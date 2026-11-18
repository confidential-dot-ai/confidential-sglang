"""Install kubectl KUBECTL_VERSION for TARGETARCH into /usr/local/bin, checked
against the sha256 published next to it on dl.k8s.io."""

import hashlib
import os
import urllib.request

base = (
    f"https://dl.k8s.io/release/{os.environ['KUBECTL_VERSION']}"
    f"/bin/linux/{os.environ['TARGETARCH']}/kubectl"
)
want = urllib.request.urlopen(base + ".sha256", timeout=60).read().decode().split()[0]
data = urllib.request.urlopen(base, timeout=300).read()
got = hashlib.sha256(data).hexdigest()
if got != want:
    raise SystemExit(f"kubectl checksum mismatch: {got} != {want}")
with open("/usr/local/bin/kubectl", "wb") as f:
    f.write(data)
os.chmod("/usr/local/bin/kubectl", 0o755)
