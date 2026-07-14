#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <nccl.h>
#include <torch/torch.h>

#include <chrono>
#include <cstdint>
#include <exception>
#include <new>
#include <string>
#include <thread>
#include <vector>

namespace {

struct OftrainNcclComm {
  ncclComm_t comm;
  int device;
  int rank;
  int world;
};

thread_local std::string last_error;

int fail(const std::string &message) {
  last_error = message;
  return -1;
}

int fail_nccl(const char *operation, ncclResult_t result) {
  return fail(std::string(operation) + ": " + ncclGetErrorString(result));
}

int abort_with_error(OftrainNcclComm *owner, const std::string &message) {
  if (owner->comm == nullptr) {
    return fail(message);
  }
  const ncclResult_t abort_result = ncclCommAbort(owner->comm);
  owner->comm = nullptr;
  if (abort_result == ncclSuccess) {
    return fail(message + "; communicator aborted");
  }
  return fail(message + "; ncclCommAbort: " + ncclGetErrorString(abort_result));
}

} // namespace

extern "C" {

const char *oftrain_nccl_last_error() { return last_error.c_str(); }

int oftrain_nccl_init_all(int world, const int *devices, void **out) {
  last_error.clear();
  if (world < 2 || devices == nullptr || out == nullptr) {
    return fail("ncclCommInitAll requires at least two devices");
  }
  try {
    std::vector<ncclComm_t> comms(static_cast<size_t>(world));
    const ncclResult_t result = ncclCommInitAll(comms.data(), world, devices);
    if (result != ncclSuccess) {
      return fail_nccl("ncclCommInitAll", result);
    }
    for (int rank = 0; rank < world; ++rank) {
      out[rank] =
          new OftrainNcclComm{comms[rank], devices[rank], rank, world};
    }
    return 0;
  } catch (const std::exception &error) {
    return fail(std::string("ncclCommInitAll wrapper: ") + error.what());
  } catch (...) {
    return fail("ncclCommInitAll wrapper: unknown exception");
  }
}

int oftrain_nccl_all_reduce(void *opaque, void *tensor_pointer,
                            uint64_t timeout_ms) {
  last_error.clear();
  if (opaque == nullptr || tensor_pointer == nullptr) {
    return fail("NCCL all-reduce received a null handle");
  }
  auto *owner = static_cast<OftrainNcclComm *>(opaque);
  auto *flat = static_cast<torch::Tensor *>(tensor_pointer);
  try {
    if (owner->comm == nullptr) {
      return fail("NCCL communicator was already aborted");
    }
    if (!flat->defined() || !flat->is_cuda() || !flat->is_contiguous() ||
        flat->scalar_type() != torch::kFloat || flat->dim() != 1) {
      return abort_with_error(
          owner,
          "NCCL all-reduce requires a contiguous 1-D CUDA float tensor");
    }
    if (flat->get_device() != owner->device) {
      return abort_with_error(
          owner, "NCCL communicator and gradient tensor devices differ");
    }

    // PyTorch guards individual CUDA operations and may restore this thread's
    // previous device afterward. Raw NCCL requires the calling thread's
    // current device to match both the communicator and stream.
    const c10::cuda::CUDAGuard device_guard(owner->device);
    const c10::cuda::CUDAStream stream =
        c10::cuda::getCurrentCUDAStream(owner->device);
    ncclResult_t result =
        ncclAllReduce(flat->data_ptr<float>(), flat->data_ptr<float>(),
                      static_cast<size_t>(flat->numel()), ncclFloat, ncclSum,
                      owner->comm, stream.stream());
    if (result != ncclSuccess) {
      return abort_with_error(
          owner, std::string("ncclAllReduce: ") + ncclGetErrorString(result));
    }

    // Queue division after the sum on the same current stream, exactly
    // matching dist.all_reduce(flat); flat /= world.
    flat->div_(owner->world);

    // Never block in cudaStreamSynchronize: a missing/misconfigured peer can
    // otherwise wedge its learner owner forever. Polling keeps the owner able
    // to abort its communicator and turn the failure into a hard train error.
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (true) {
      ncclResult_t asynchronous = ncclSuccess;
      result = ncclCommGetAsyncError(owner->comm, &asynchronous);
      if (result != ncclSuccess) {
        return abort_with_error(
            owner, std::string("ncclCommGetAsyncError: ") +
                       ncclGetErrorString(result));
      }
      if (asynchronous != ncclSuccess) {
        return abort_with_error(
            owner, std::string("NCCL asynchronous collective: ") +
                       ncclGetErrorString(asynchronous));
      }
      if (stream.query()) {
        return 0;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        return abort_with_error(
            owner, "NCCL all-reduce timed out after " +
                       std::to_string(timeout_ms) + "ms on rank " +
                       std::to_string(owner->rank) + " cuda:" +
                       std::to_string(owner->device));
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } catch (const std::exception &error) {
    return abort_with_error(
        owner, std::string("NCCL all-reduce wrapper: ") + error.what());
  } catch (...) {
    return abort_with_error(owner,
                            "NCCL all-reduce wrapper: unknown exception");
  }
}

int oftrain_nccl_destroy(void *opaque) {
  last_error.clear();
  if (opaque == nullptr) {
    return 0;
  }
  auto *owner = static_cast<OftrainNcclComm *>(opaque);
  const ncclResult_t result =
      owner->comm == nullptr ? ncclSuccess : ncclCommDestroy(owner->comm);
  delete owner;
  return result == ncclSuccess ? 0 : fail_nccl("ncclCommDestroy", result);
}

} // extern "C"
