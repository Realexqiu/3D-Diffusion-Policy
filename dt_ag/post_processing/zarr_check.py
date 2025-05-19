#!/usr/bin/env python3
"""
Zarr Episode Viewer - First & Last Frame Visualization
======================================================
"""

from pathlib import Path
import numpy as np
import zarr
import open3d as o3d
import matplotlib.pyplot as plt
from rich.console import Console

console = Console()

class ZarrFrameVisualizer:
    def __init__(self, zarr_path: Path):
        if not zarr_path.exists():
            raise FileNotFoundError(f"Zarr store not found: {zarr_path}")
        self.root = zarr.open(zarr_path, mode="r")

        # pick first episode key
        eps = sorted(k for k in self.root.keys() if k.startswith("episode_"))
        if not eps:
            raise KeyError("No 'episode_##' groups found in Zarr store.")
        self.episode_key = eps[0]
        self.ep = self.root[self.episode_key]

        console.print(f"[green]Loaded episode[/] [bold]{self.episode_key}[/]")

        # required datasets for RGB/depth
        for cam in ("rs", "zed"):
            for dt in ("rgb", "depth"):
                key = f"{cam}_{dt}"
                if key not in self.ep:
                    raise KeyError(f"Missing dataset '{key}' in episode.")

        # optional: zed point clouds
        if "zed_pcd" not in self.ep:
            console.print("[yellow]Warning: 'zed_pcd' not found—skipping point-cloud viz/save[/]")
            self.has_pcd = False
        else:
            self.has_pcd = True

        # read length
        if "length" in self.ep.attrs:
            self.length = int(self.ep.attrs["length"])
        else:
            # infer from any rgb
            self.length = self.ep["rs_rgb"].shape[0]

    def print_structure(self):
        console.print("\n[bold]Episode data structure:[/]")
        for k in self.ep.keys():
            console.print(f" • {k}: shape={self.ep[k].shape}, dtype={self.ep[k].dtype}")
        if self.ep.attrs:
            console.print("Attributes:")
            for a, v in self.ep.attrs.items():
                console.print(f"   – {a}: {v}")
        console.print()

    def visualize_frames(self, output_dir: Path):
        output_dir.mkdir(exist_ok=True, parents=True)
        for idx, label in [(0, "first"), (self.length-1, "last")]:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            for row, cam in enumerate(("rs", "zed")):
                rgb = self.ep[f"{cam}_rgb"][idx]       # BGR uint8
                depth = self.ep[f"{cam}_depth"][idx]   # float32

                # RGB
                axes[row,0].imshow(rgb[..., ::-1])
                axes[row,0].set_title(f"{cam.upper()} RGB (frame {idx})")
                axes[row,0].axis("off")

                # Depth
                im = axes[row,1].imshow(depth, cmap="viridis")
                axes[row,1].set_title(f"{cam.upper()} Depth (frame {idx})")
                axes[row,1].axis("off")
                fig.colorbar(im, ax=axes[row,1], label="meters")

            plt.suptitle(f"{label.title()} Frame (idx={idx})")
            plt.tight_layout()
            out_file = output_dir / f"{label}_frames.png"
            plt.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close(fig)
            console.print(f"[green]Saved[/] {out_file}")

    def visualize_pcd(self):
        if not self.has_pcd:
            return

        for idx, label in [(0, "First"), (self.length-1, "Last")]:
            pc = self.ep["zed_pcd"][idx]  # (N,6)
            if pc.size == 0 or not np.any(pc):
                console.print(f"[yellow]zed_pcd at idx={idx} is empty; skipping[/]")
                continue

            pcd = o3d.geometry.PointCloud()
            pts = pc[:, :3]
            cols = pc[:, 3:6] if pc.shape[1] >= 6 else np.ones_like(pts)*0.5
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(cols)

            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=f"{label} ZED PointCloud", width=800, height=600)
            vis.add_geometry(pcd)
            vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
            opt = vis.get_render_option()
            opt.background_color = np.array([0.1, 0.1, 0.1])
            opt.point_size = 2.0

            console.print(f"[green]Opening[/] {label} point cloud (idx={idx}). Close window to continue.")
            vis.run()
            vis.destroy_window()

    def save_ply(self, output_dir: Path):
        if not self.has_pcd:
            return

        output_dir.mkdir(exist_ok=True, parents=True)
        for idx, label in [(0, "first"), (self.length-1, "last")]:
            pc = self.ep["zed_pcd"][idx]
            if pc.size == 0 or not np.any(pc):
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
            if pc.shape[1] >= 6:
                pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:6])

            out_file = output_dir / f"{label}_zed_pcd.ply"
            o3d.io.write_point_cloud(str(out_file), pcd)
            console.print(f"[green]Wrote PLY[/] {out_file}")

def main():
    # Use a hardcoded Zarr path here; update the path as needed.
    store = Path("/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data/3d_strawberry_dt/3d_strawberry_dt_50_zarr")
    viz = ZarrFrameVisualizer(store)

    viz.print_structure()

    out_dir = Path("frame_visualization")
    viz.visualize_frames(out_dir)
    viz.visualize_pcd()
    viz.save_ply(out_dir)

    console.print("\n[bold green]All done![/] Check 'frame_visualization/' for outputs.")

if __name__ == "__main__":
    main()
