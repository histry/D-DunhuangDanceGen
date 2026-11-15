EDGE is an external pinned runtime, not a fork. Use official commit
17c3428669ed6733edd9d8c66f7dc62060b8e46d from https://github.com/Stanford-TML/EDGE.
The server configuration reuses its existing clean checkout. Alternatively clone
into third_party/EDGE and check out that exact commit. Follow upstream environment
installation. No pretrained weights or restricted motion data are redistributed.


R22 Lodge Local runtime: official source `li-ronghui/LODGE` pinned to `f38642ad67bc3402444f5f914010a911e5bd5c89`, official README-linked Local epoch299 checkpoint. Source, Normalizer, skeleton, matching YAML and weights are individually SHA-bound in ignored `inputs/lodge_r22/runtime_manifest.json`. R22 does not redistribute the full upstream checkout or pretrained model in this repository; see `docs/R22_PROTOCOL.md` for acquisition, task and incomplete-skeleton boundaries.
