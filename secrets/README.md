# Secrets

Encrypted-at-rest via [SOPS](https://github.com/getsops/sops) + [age](https://github.com/FiloSottile/age) (see `../.sops.yaml`). Only `*.enc.yaml` files here are ever committed - everything else in this directory is gitignored, so an accidental plaintext decrypt-in-place never gets pushed.

## Adding a secret

```bash
echo 'some_key: "value"' > secrets/whatever.enc.yaml
sops --encrypt --in-place secrets/whatever.enc.yaml
git add secrets/whatever.enc.yaml
```

## Reading secrets (in a shell session, CI, or a training pod)

```bash
export SOPS_AGE_KEY="AGE-SECRET-KEY-..."   # or SOPS_AGE_KEY_FILE=/path/to/key
source scripts/load_secrets.sh
echo "$RUNPOD_API_KEY"   # yaml key `runpod_api_key` -> env var `RUNPOD_API_KEY`
```

The age **private** key is deliberately never committed anywhere in this repo - it must be supplied out-of-band each session (e.g. as a Cursor background-agent environment secret, or from wherever you keep it - a password manager, your homelab's own secret store, etc.). Losing it means re-encrypting every `*.enc.yaml` under a freshly generated key and re-distributing secrets from scratch.

## Current secrets

- `runpod.enc.yaml`: `runpod_api_key` - a dedicated `cursor-agent`-named RunPod API key (created via `createApiKey`, scoped separately from any personal key so it can be revoked independently).
- `share_skg.enc.yaml`: `share_skg_url` / `share_skg_username` / `share_skg_password` / `share_skg_token` → env `SHARE_SKG_*` — Zipline account on https://share.skg.gg for agent clip uploads (`cursor` user).
- `hf.enc.yaml`: `hf_token` → env `HF_TOKEN` - Hugging Face **write** token used by training pods / HF sync workers to upload checkpoints. Prefer a dedicated `cursor-agent` token from https://huggingface.co/settings/tokens so it can be revoked independently of your personal login; re-encrypt after rotating.
