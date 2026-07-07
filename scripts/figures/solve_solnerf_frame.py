"""Kabsch solve: SOL-NeRF paper normals -> our lwp normals (raw frame)."""
import sys
sys.path.insert(0, "/workspace/phd/code/neusky/scripts/figures")
import numpy as np
import torch
import cv2
from _common import load_model, render_view

CROP = (0.0078, 0.0047, 0.9922, 0.8246)

_, pipeline, _, _ = load_model("lwp", device="cuda:0")
model = pipeline.model
model.eval()
cameras = pipeline.datamanager.train_dataset.cameras
outputs = render_view(model, cameras, 234, "cuda:0")
n_ours = outputs["normal"].cpu().numpy()          # raw, unit, ours frame
acc = outputs["accumulation"].cpu().numpy()[..., 0]

h, w = n_ours.shape[:2]
x0, y0, x1, y1 = CROP
n_ours = n_ours[int(y0*h):int(y1*h), int(x0*w):int(x1*w)]
acc = acc[int(y0*h):int(y1*h), int(x0*w):int(x1*w)]

sol = cv2.imread("/workspace/phd/code/neusky/scripts/figures/assets/sol_nerf_site3_lwp_normals_from_paper.jpg")
sol = sol.astype(np.float32)[..., ::-1] / 255.0
sol = cv2.resize(sol, (n_ours.shape[1], n_ours.shape[0]))
n_sol = sol * 2 - 1

norm_sol = np.linalg.norm(n_sol, axis=-1)
valid = (acc > 0.7) & (norm_sol > 0.6) & (norm_sol < 1.4)
# drop the sky band: SOL sky is whitish -> norm ~ sqrt(3); already cut by <1.4
P = n_sol[valid]; P = P / np.linalg.norm(P, axis=1, keepdims=True)
Q = n_ours[valid]; Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
print("correspondences:", P.shape[0])
H = P.T @ Q
U, S, Vt = np.linalg.svd(H)
M = (Vt.T @ U.T)   # allows improper if conventions differ
res = np.linalg.norm(P @ M.T - Q, axis=1).mean()
print("det:", round(float(np.linalg.det(M)), 3), "mean residual:", round(float(res), 4))
print("M_sol_to_ours =", np.round(M, 4).tolist())
OURS_TO_FEGR = np.array([[-1,0,0],[0,0,1],[0,-1,0]], np.float32)
MF = OURS_TO_FEGR @ M
print("M_sol_to_FEGRdisplay =", np.round(MF, 4).tolist())

# snap test: constrain to up-preserving improper transform (flip + heading)
best = None
for flip in (np.diag([-1.0, 1.0, 1.0]), np.eye(3)):
    for deg in np.arange(0, 360, 0.5):
        c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        Msnap = Ry @ flip
        r = np.linalg.norm(P @ (OURS_TO_FEGR @ np.linalg.inv(OURS_TO_FEGR) @ Msnap).T - (Q @ np.linalg.inv(OURS_TO_FEGR).T if False else Q), axis=1).mean() if False else None
        # target is ours-frame Q; snap candidate maps sol->ours directly
        r = np.linalg.norm(P @ (np.linalg.inv(OURS_TO_FEGR) @ Msnap).T - Q, axis=1).mean()
        if best is None or r < best[0]:
            best = (float(r), float(deg), float(np.linalg.det(Msnap)))
print("snapped (up-exact, FEGR-display flip+heading): residual",
      round(best[0], 4), "deg", best[1], "det", best[2])
print("free-solve residual:", round(float(res), 4))
