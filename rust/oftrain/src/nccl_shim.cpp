#include <c10/cuda/CUDAStream.h>
#include <nccl.h>
#include <torch/torch.h>

#include <cstdint>
#include <exception>
#include <new>
#include <string>
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

int oftrain_nccl_all_reduce(void *opaque, void *tensor_pointer) {
  last_error.clear();
  if (opaque == nullptr || tensor_pointer == nullptr) {
    return fail("NCCL all-reduce received a null handle");
  }
  auto *owner = static_cast<OftrainNcclComm *>(opaque);
  auto *flat = static_cast<torch::Tensor *>(tensor_pointer);
  try {
    if (!flat->defined() || !flat->is_cuda() || !flat->is_contiguous() ||
        flat->scalar_type() != torch::kFloat || flat->dim() != 1) {
      return fail(
          "NCCL all-reduce requires a contiguous 1-D CUDA float tensor");
    }
    if (flat->get_device() != owner->device) {
      return fail("NCCL communicator and gradient tensor devices differ");
    }

    // c10's current stream is thread-local. The caller guarantees that this
    // communicator and every tensor operation stay on the learner owner.
    const c10::cuda::CUDAStream stream =
        c10::cuda::getCurrentCUDAStream(owner->device);
    ncclResult_t result =
        ncclAllReduce(flat->data_ptr<float>(), flat->data_ptr<float>(),
                      static_cast<size_t>(flat->numel()), ncclFloat, ncclSum,
                      owner->comm, stream.stream());
    if (result != ncclSuccess) {
      return fail_nccl("ncclAllReduce", result);
    }

    // Queue division after the sum on the same current stream, exactly
    // matching dist.all_reduce(flat); flat /= world.
    flat->div_(owner->world);
    stream.synchronize();
    ncclResult_t asynchronous = ncclSuccess;
    result = ncclCommGetAsyncError(owner->comm, &asynchronous);
    if (result != ncclSuccess) {
      return fail_nccl("ncclCommGetAsyncError", result);
    }
    if (asynchronous != ncclSuccess) {
      return fail_nccl("NCCL asynchronous collective", asynchronous);
    }
    return 0;
  } catch (const std::exception &error) {
    return fail(std::string("NCCL all-reduce wrapper: ") + error.what());
  } catch (...) {
    return fail("NCCL all-reduce wrapper: unknown exception");
  }
}

int oftrain_nccl_destroy(void *opaque) {
  last_error.clear();
  if (opaque == nullptr) {
    return 0;
  }
  auto *owner = static_cast<OftrainNcclComm *>(opaque);
  const ncclResult_t result = ncclCommDestroy(owner->comm);
  delete owner;
  return result == ncclSuccess ? 0 : fail_nccl("ncclCommDestroy", result);
}

} // extern "C"
