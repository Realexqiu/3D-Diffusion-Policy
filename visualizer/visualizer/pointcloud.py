#!/usr/bin/env python3
"""
3D Point Cloud Visualizer
=========================

A lightweight Flask+Plotly server that visualizes 3D point clouds.
The latest point cloud is stored in memory and displayed with auto-refresh.
Multiple viewing angles are supported.
"""

from flask import Flask, render_template_string, redirect, url_for
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
import logging
import threading
import sys

class Visualizer:
    """
    A simple Flask + Plotly server that always displays the latest 3D point cloud.
    Call .run() once in a background thread, and update_pcd() to show new data.
    
    Features:
    - Auto-refreshing visualization
    - Multiple preset camera angles
    - Thread-safe updates
    - Minimal logging output
    """
    def __init__(self, camera_position='isometric'):
        self.app = Flask(__name__)
        self.latest_pcd = None  # holds the most recent point cloud data
        self._lock = threading.Lock()  # Add thread safety
        
        # Set default camera positions
        self.camera_positions = {
            'side': dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=0.5)
            ),
            'top': dict(
                up=dict(x=0, y=-1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0, y=0, z=2.5)
            ),
            'front': dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0, y=-2.0, z=0.5)
            ),
            'isometric': dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.25, y=1.25, z=1.25)
            ),
            'robot': dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0.5, y=-1.0, z=0.8)
            )
        }
        
        # Set the camera position (can be a string key or a custom dict)
        if isinstance(camera_position, str):
            self.camera_position = self.camera_positions.get(
                camera_position, self.camera_positions['isometric'])
        else:
            self.camera_position = camera_position
            
        # Disable Flask logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        self.app.logger.setLevel(logging.ERROR)
        import flask.cli
        flask.cli.show_server_banner = lambda *args, **kwargs: None

        # HTML template with auto-refresh and view controls
        self.template = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>3D Point Cloud Visualizer</title>
            <meta http-equiv="refresh" content="3">
            <style>
                body { margin: 0; padding: 0; font-family: Arial, sans-serif; }
                .controls {
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    z-index: 1000;
                    background: rgba(255,255,255,0.8);
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.2);
                }
                .controls a {
                    display: inline-block;
                    margin: 0 5px;
                    padding: 5px 10px;
                    background: #4CAF50;
                    color: white;
                    text-decoration: none;
                    border-radius: 3px;
                }
                .controls a:hover {
                    background: #45a049;
                }
                #plot {
                    width: 100vw;
                    height: 100vh;
                }
            </style>
        </head>
        <body>
            <div class="controls">
                <span>View: </span>
                <a href="/view/isometric">Isometric</a>
                <a href="/view/top">Top</a>
                <a href="/view/front">Front</a>
                <a href="/view/side">Side</a>
                <a href="/view/robot">Robot</a>
            </div>
            <div id="plot">{{ plot_html|safe }}</div>
        </body>
        </html>
        '''

        @self.app.route('/')
        def index():
            with self._lock:
                pcd_data = self.latest_pcd
            
            if pcd_data is None:
                return render_template_string(self.template, plot_html="<h3>No point cloud streamed yet.</h3>")
            
            try:
                pc = pcd_data
                
                # Make sure we have at least XYZ coordinates
                if pc.shape[1] < 3:
                    return render_template_string(self.template, 
                                                 plot_html="<h3>Error: Point cloud must have at least 3 dimensions (x,y,z)</h3>")
                
                # Extract coordinates
                x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
                
                # Handle colors - if provided directly in the point cloud data
                if pc.shape[1] >= 6:  # Assuming XYZ + RGB format
                    r, g, b = pc[:, 3], pc[:, 4], pc[:, 5]
                    colors = [
                        f"rgb({int(min(max(r_val*255, 0), 255))},{int(min(max(g_val*255, 0), 255))},{int(min(max(b_val*255, 0), 255))})"
                        for r_val, g_val, b_val in zip(r, g, b)
                    ]
                else:
                    # Generate colors based on z-height if no colors provided
                    z_range = z.max() - z.min() if z.max() != z.min() else 1.0
                    norm_z = (z - z.min()) / z_range
                    colors = [
                        f"rgb({int(val*255)},0,{int((1-val)*255)})"
                        for val in norm_z
                    ]
                
                # Create the 3D scatter plot
                trace = go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(
                        size=3,
                        opacity=0.8,
                        color=colors
                    )
                )
                
                # Create and configure the figure layout
                fig = go.Figure(data=[trace])
                
                # Add coordinate axes for reference
                axis_size = 0.2
                max_range = max(
                    x.max() - x.min(), 
                    y.max() - y.min(), 
                    z.max() - z.min()
                )
                
                # X axis (red)
                fig.add_trace(go.Scatter3d(
                    x=[0, axis_size * max_range], y=[0, 0], z=[0, 0],
                    mode='lines',
                    line=dict(color='red', width=4),
                    showlegend=False
                ))
                
                # Y axis (green)
                fig.add_trace(go.Scatter3d(
                    x=[0, 0], y=[0, axis_size * max_range], z=[0, 0],
                    mode='lines',
                    line=dict(color='green', width=4),
                    showlegend=False
                ))
                
                # Z axis (blue)
                fig.add_trace(go.Scatter3d(
                    x=[0, 0], y=[0, 0], z=[0, axis_size * max_range],
                    mode='lines',
                    line=dict(color='blue', width=4),
                    showlegend=False
                ))
                
                # Update layout with camera position
                fig.update_layout(
                    margin=dict(l=0, r=0, b=0, t=30),
                    scene=dict(
                        xaxis=dict(showgrid=True, zeroline=True, title='X'),
                        yaxis=dict(showgrid=True, zeroline=True, title='Y'),
                        zaxis=dict(showgrid=True, zeroline=True, title='Z'),
                        aspectmode='data',
                        camera=self.camera_position
                    ),
                    title=dict(
                        text=f"Point Cloud - {pc.shape[0]} points", 
                        y=0.95,
                        x=0.5,
                        xanchor='center',
                        yanchor='top'
                    )
                )
                
                # Convert to HTML but embed in our auto-refreshing template
                plot_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
                return render_template_string(self.template, plot_html=plot_html)
                
            except Exception as e:
                error_msg = f"Error visualizing point cloud: {str(e)}"
                return render_template_string(self.template, plot_html=f"<h3>{error_msg}</h3>")
        
        # Route to change the view
        @self.app.route('/view/<viewname>')
        def change_view(viewname):
            if viewname in self.camera_positions:
                self.camera_position = self.camera_positions[viewname]
                return redirect('/')
            else:
                available_views = ', '.join(self.camera_positions.keys())
                return f"Unknown view '{viewname}'. Available views: {available_views}"

    def update_pcd(self, pointcloud):
        """
        Thread-safe method to update the in-memory point cloud.
        
        Args:
            pointcloud: numpy array of shape (N, D) where D >= 3
                        first 3 dimensions must be XYZ coordinates
                        dimensions 4-6 can be RGB values (0-1 range)
        """
        if not isinstance(pointcloud, np.ndarray):
            return
            
        if pointcloud.shape[0] == 0:
            return
            
        if pointcloud.shape[1] < 3:
            return
        
        # Thread-safe update of the point cloud    
        with self._lock:
            self.latest_pcd = pointcloud.copy()  # Make a copy to avoid any reference issues

    def run(
        self,
        host='0.0.0.0',
        port=8050,
        debug=False,
        use_reloader=False
    ):
        """
        Start the Flask server. This call is blocking, so
        run it in a daemon thread from your main node.
        
        Args:
            host: Host to bind the server to
            port: Port to bind the server to
            debug: Enable Flask debug mode
            use_reloader: Enable Flask auto-reloader (not needed in a thread)
        """
        print(f"Starting point cloud visualizer on http://{host}:{port}")
        self.app.run(
            host=host,
            port=port,
            debug=debug,
            use_reloader=use_reloader
        )


# Example usage when run directly
if __name__ == '__main__':
    import time
    
    # Create a test point cloud - a spiral with colors
    t = np.linspace(0, 10*np.pi, 1000)
    x = np.sin(t)
    y = np.cos(t)
    z = t / 10
    
    # Generate r,g,b colors that transition with height
    r = np.clip(np.sin(t/5), 0, 1)
    g = np.clip(np.cos(t/3), 0, 1)
    b = np.clip(z/3, 0, 1)
    
    points = np.column_stack([x, y, z, r, g, b])
    
    # Create and run visualizer
    viz = Visualizer(camera_position='isometric')
    viz.update_pcd(points)
    
    # Start the server in the main thread (for demonstration)
    print("Point Cloud Visualizer example running...")
    print("View in your browser at http://localhost:8050")
    print("Press Ctrl+C to exit")
    
    # Start the Flask app
    viz.run(debug=False, use_reloader=False)