# Reliable Purffle operation on Termux

This keeps Purffle's existing local setup and settings. It adds a small supervisor that restarts the autopilot process after a crash and an optional Termux:Boot hook to start it after Android reboots.

## Install once

From the existing repository in Termux (do not clone a second copy):

```bash
cd ~/purffle-shorts-cloud
git status --short --branch
git pull --ff-only origin main
bash termux/install-service.sh
```

If Git reports local changes or refuses the fast-forward, stop and inspect the message instead of resetting or discarding work. The installer may install the Termux `util-linux` package to provide `flock`, which prevents two supervisors from running at once.

The installer does not edit `.env`, credentials, tokens, or enable uploading. The worker runs the same `python -m purffle_shorts run` command using the repository's existing settings.

## Commands

```bash
bash ~/purffle-shorts-cloud/termux/purffle-service.sh status
bash ~/purffle-shorts-cloud/termux/purffle-service.sh logs
bash ~/purffle-shorts-cloud/termux/purffle-service.sh stop
bash ~/purffle-shorts-cloud/termux/purffle-service.sh start
bash ~/purffle-shorts-cloud/termux/purffle-service.sh restart
```

Logs are saved at `data/logs/purffle-service.log`; the service PID and lock are in `data/service/`. These runtime files are ignored by Git with the rest of `data/`.

## Automatic startup after a reboot

1. Install **Termux:Boot** from the same trusted source used for Termux.
2. Open Termux:Boot once.
3. Run `bash termux/install-service.sh` in the repository. It creates `~/.termux/boot/purffle.sh`.
4. In Android Settings, set Termux battery usage to **Unrestricted** (wording varies by Android version). Allow its persistent notification.
5. Test by stopping and starting the service before attempting a phone reboot.

Termux:Boot is a separate Android add-on. The repository cannot install or authorize the app itself. Android/OEM battery management can still interrupt background processes, so this is best-effort recovery, not a guarantee of 24/7 uptime.

## What recovery guarantees—and what it doesn't

- The supervisor relaunches the autopilot process after an unexpected exit, with a 20-second pause.
- Purffle's SQLite history and existing Studio queue snapshots are stored on disk and survive process restarts.
- A render interrupted before its result is saved may need to be rendered again. The supervisor does not try to resume FFmpeg halfway through a file.
- If Android kills Termux completely, only Termux:Boot can restart the service after reboot; install and open that add-on and set battery permissions.
- Never put `.env`, `credentials.json`, `token.json`, or other secrets in GitHub. This setup doesn't upload them.

## Safe YouTube upload recovery

A network interruption at exactly the wrong moment can occur after YouTube accepts a video but before Purffle saves the returned video ID. For safety, Purffle marks an in-progress upload in SQLite. If the process dies in that narrow window, it will leave the record in an uncertain state rather than blindly uploading it again. Check YouTube Studio for the title before deciding what to do next. This avoids automatic duplicates, although it may require a manual check in that rare case.

## Cloudflare

Cloudflare is **not required** for this local Termux watchdog/recovery setup. It is only useful later if Studio should be publicly hosted or if video files should be served from remote object storage.
