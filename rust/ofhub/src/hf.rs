//! Hugging Face Hub helpers. Write paths fail loud without `HF_TOKEN`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use hf_hub::repository::{AddSource, RepoTreeEntry, RepoTypeModel};
use hf_hub::HFClient;

use crate::paths::{policy_dir, revision_path, AE_REPO, POLICY_REPO};
use crate::util::{hf_policy_paths, LEGACY_POLICY_RUNS};

pub fn require_hf_token() -> Result<String> {
    match env::var("HF_TOKEN") {
        Ok(t) if !t.trim().is_empty() => Ok(t),
        _ => bail!("HF_TOKEN is required for Hugging Face write/auth paths (fail loud)"),
    }
}

pub fn client_with_optional_token() -> Result<HFClient> {
    if let Ok(token) = env::var("HF_TOKEN") {
        if !token.trim().is_empty() {
            return Ok(HFClient::builder().token(token).build()?);
        }
    }
    Ok(HFClient::new()?)
}

pub fn client_requiring_token() -> Result<HFClient> {
    let token = require_hf_token()?;
    Ok(HFClient::builder().token(token).build()?)
}

pub fn split_repo(repo_id: &str) -> Result<(String, String)> {
    let (owner, name) = repo_id
        .split_once('/')
        .with_context(|| format!("invalid repo id {repo_id:?}"))?;
    Ok((owner.to_string(), name.to_string()))
}

pub async fn download_to(
    client: &HFClient,
    repo_id: &str,
    remote: &str,
    dest: &Path,
) -> Result<PathBuf> {
    let (owner, name) = split_repo(repo_id)?;
    let repo = client.model(owner, name);
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    let local_dir = dest
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let staged = repo
        .download_file()
        .filename(remote)
        .local_dir(local_dir)
        .send()
        .await
        .with_context(|| format!("download {repo_id}/{remote}"))?;
    if staged != dest {
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&staged, dest)
            .with_context(|| format!("copy {} -> {}", staged.display(), dest.display()))?;
    }
    Ok(dest.to_path_buf())
}

pub async fn upload_file(
    client: &HFClient,
    repo_id: &str,
    local: &Path,
    remote: &str,
    commit_message: &str,
) -> Result<()> {
    let _ = require_hf_token()?;
    let (owner, name) = split_repo(repo_id)?;
    let repo = client.model(owner, name);
    repo.upload_file()
        .source(AddSource::file(local))
        .path_in_repo(remote)
        .commit_message(commit_message)
        .send()
        .await
        .with_context(|| format!("upload {local:?} -> {repo_id}/{remote}"))?;
    Ok(())
}

pub async fn ensure_repo(client: &HFClient, repo_id: &str) -> Result<()> {
    let _ = require_hf_token()?;
    client
        .create_repository()
        .repo_id(repo_id)
        .repo_type(RepoTypeModel)
        .exist_ok(true)
        .send()
        .await
        .with_context(|| format!("create_repository {repo_id}"))?;
    Ok(())
}

pub async fn whoami(client: &HFClient) -> Result<String> {
    let user = client.whoami().send().await?;
    Ok(user.username)
}

pub async fn file_revision(client: &HFClient, repo_id: &str, path: &str) -> Result<String> {
    let (owner, name) = split_repo(repo_id)?;
    let repo = client.model(owner, name);
    match repo
        .get_paths_info()
        .paths(vec![path.to_string()])
        .send()
        .await
    {
        Ok(entries) => {
            for entry in entries {
                if let RepoTreeEntry::File { oid, .. } = entry {
                    return Ok(oid);
                }
            }
        }
        Err(_) => {}
    }
    let meta = repo
        .get_file_metadata()
        .filepath(path)
        .send()
        .await
        .with_context(|| format!("metadata for {repo_id}/{path}"))?;
    Ok(meta.etag)
}

pub async fn policy_revision(client: &HFClient, run_name: &str) -> Result<String> {
    let (weights, _) = hf_policy_paths(run_name);
    file_revision(client, POLICY_REPO, &weights).await
}

pub fn is_legacy_policy_run(run_name: &str) -> bool {
    LEGACY_POLICY_RUNS.contains(&run_name)
}

pub async fn ensure_policy(client: &HFClient, run_name: &str) -> Result<PathBuf> {
    let (weights_path, state_path) = hf_policy_paths(run_name);
    let dest = policy_dir().join(&weights_path);
    let state_dest = state_path.as_ref().map(|p| policy_dir().join(p));
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }

    let cache_complete = dest.exists()
        && state_dest
            .as_ref()
            .map(|p| p.exists())
            .unwrap_or(true);

    // Prefer a short Hub probe; if Hub hangs/unreachable, keep serving local weights.
    let revision = match tokio::time::timeout(
        std::time::Duration::from_secs(20),
        policy_revision(client, run_name),
    )
    .await
    {
        Ok(Ok(r)) => r,
        Ok(Err(e)) if cache_complete => {
            eprintln!(
                "[ofhub] warning: policy revision check failed ({e}); using local {}",
                dest.display()
            );
            return Ok(dest);
        }
        Err(_) if cache_complete => {
            eprintln!(
                "[ofhub] warning: policy revision check timed out; using local {}",
                dest.display()
            );
            return Ok(dest);
        }
        Ok(Err(e)) if !is_legacy_policy_run(run_name) => {
            eprintln!(
                "[ofhub] warning: {run_name}/latest.safetensors unavailable ({e}); \
                 trying policy.pt fallback"
            );
            let fallback = format!("{run_name}/policy.pt");
            let dest_pt = policy_dir().join(&fallback);
            if dest_pt.exists() {
                return Ok(dest_pt);
            }
            download_to(client, POLICY_REPO, &fallback, &dest_pt).await?;
            if let Ok(Ok(rev)) = tokio::time::timeout(
                std::time::Duration::from_secs(20),
                file_revision(client, POLICY_REPO, &fallback),
            )
            .await
            {
                fs::write(revision_path(), &rev)?;
            }
            return Ok(dest_pt);
        }
        Ok(Err(e)) => return Err(e),
        Err(_) => bail!("policy revision check timed out and no local cache for {run_name}"),
    };

    if cache_complete && revision_path().exists() {
        let local = fs::read_to_string(revision_path()).unwrap_or_default();
        if local.trim() == revision {
            return Ok(dest);
        }
    }

    download_to(client, POLICY_REPO, &weights_path, &dest).await?;
    if let (Some(remote_state), Some(local_state)) = (state_path.as_ref(), state_dest.as_ref()) {
        download_to(client, POLICY_REPO, remote_state, local_state).await?;
    }
    fs::write(revision_path(), &revision)?;
    Ok(dest)
}

pub async fn ensure_ae_encoder(
    client: &HFClient,
    remote_name: &str,
    dest: &Path,
) -> Result<PathBuf> {
    if dest.exists() {
        return Ok(dest.to_path_buf());
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    download_to(client, AE_REPO, remote_name, dest).await
}

pub fn default_policy_repo() -> String {
    env::var("HF_REPO_ID").unwrap_or_else(|_| POLICY_REPO.to_string())
}
