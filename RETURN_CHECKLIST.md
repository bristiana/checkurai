# Before returning the Intel laptop

Challenge terms: delete your files before returning the kit; cloud instances may shut down after the event.

## 1. Back up first
- [ ] Datasets (recorded episodes) copied off the laptop (Hugging Face Hub / drive)
- [ ] Trained models + OpenVINO exports copied off the laptop and off the cloud GPU instances
- [ ] Anomalib training images (`snapshots/`) backed up
- [ ] Final `git push` done

## 2. Revoke access
- [ ] Delete the deploy key: https://github.com/bristiana/checkurai/settings/keys -> "Intel laptop (CheckurAI station)"
- [ ] Remove password-free sudo: `sudo rm /etc/sudoers.d/90-checkurai`
- [ ] Remove deploy key files: `rm ~/.ssh/checkurai_deploy ~/.ssh/checkurai_deploy.pub` and the `Host github-checkurai` block in `~/.ssh/config`
- [ ] Remove any Hugging Face / cloud tokens: `huggingface-cli logout`, check `~/.cache/huggingface/token`, Studio `.env` files
- [ ] Team SSH keys removed from `~/.ssh/authorized_keys` (Marius)

## 3. Clean up
- [ ] Stop the camera dashboard: `pkill -f "python tools/camdash"`
- [ ] Delete `~/checkurai` and any datasets/models in `~/.cache/huggingface/lerobot`
