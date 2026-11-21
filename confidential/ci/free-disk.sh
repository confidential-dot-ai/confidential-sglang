#!/usr/bin/env bash
# Make room on a GitHub-hosted ubuntu-24.04 runner for one engine build
# (~20 GB image, more in BuildKit's state): drop preinstalled toolchains, and
# move Docker's data root and the build's scratch space to the larger /mnt.
# As confidential-inference's release-images.yml does for its sglang image.
set -euo pipefail
echo "--- disk before ---"; df -h / /mnt || true
sudo rm -rf -- /usr/share/dotnet /usr/local/lib/android /opt/ghc /opt/hostedtoolcache \
  /usr/local/.ghcup /usr/share/swift /usr/local/share/powershell /usr/local/julia* \
  /usr/local/lib/heroku /opt/az /usr/local/share/boost /usr/share/miniconda \
  /usr/lib/jvm /opt/microsoft /usr/local/share/chromium /usr/local/lib/node_modules \
  /usr/local/aws-cli /usr/share/google-cloud-sdk /usr/local/games \
  "${AGENT_TOOLSDIRECTORY:-/opt/hostedtoolcache}" 2>/dev/null || true
sudo apt-get purge -y --auto-remove \
  '^ghc-.*' '^dotnet-.*' '^llvm-.*' '^mysql-.*' '^php.*' azure-cli \
  google-cloud-cli google-chrome-stable firefox powershell mono-devel \
  libgl1-mesa-dri '^temurin-.*' >/dev/null 2>&1 || true
sudo apt-get clean
sudo swapoff -a || true
sudo rm -f /mnt/swapfile /swapfile 2>/dev/null || true
sudo service docker stop
sudo service containerd stop
sudo mkdir -p /mnt/docker-data /mnt/containerd /mnt/tmp /mnt/oci /mnt/record
sudo chown "$(id -u):$(id -g)" /mnt/tmp /mnt/oci /mnt/record
# The containerd image store: the classic store's docker load cannot read an
# OCI archive with zstd layers (run 44874491789: "open .../blobs/json"). Its
# images live in containerd's root, so that moves to /mnt too.
printf '{"data-root":"/mnt/docker-data","features":{"containerd-snapshotter":true}}\n' \
  | sudo tee /etc/docker/daemon.json >/dev/null
sudo rm -rf /var/lib/docker /var/lib/containerd
sudo mkdir -p /var/lib/containerd
sudo mount --bind /mnt/containerd /var/lib/containerd
sudo service containerd start
sudo service docker start
echo "--- disk after ---"; df -h / /mnt
test "$(docker info --format '{{.DockerRootDir}}')" = /mnt/docker-data
docker info --format '{{json .DriverStatus}}' | grep -q io.containerd.snapshotter.v1
