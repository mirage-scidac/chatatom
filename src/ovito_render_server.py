# Copyright © 2026, UChicago Argonne, LLC All Rights Reserved
# Full license accessible at https://github.com/mirage-scidac/chatatom/blob/main/LICENSE

"""
Standalone OVITO render server.

Launched as a subprocess by tools.py. Reads JSON render requests from stdin,
writes JSON results (base64 PNG) to stdout. Keeps OVITO pipeline cached across
frames of the same dump file so import_file() is only called once per file.

Each simulation gets its own subprocess instance — avoids semaphore leaks
that crash the Flask process when import_file() is called multiple times.
"""
import sys
import json
import base64
import os
import tempfile


def apply_basic_viz(pipeline):
    """Color by Particle Type (always present). Fall back silently if unavailable."""
    try:
        from ovito.modifiers import ColorCodingModifier
        pipeline.modifiers.append(ColorCodingModifier(property='Particle Type'))
    except Exception:
        pass  # render with default colors if modifier fails


def main():
    import ovito
    from ovito.io import import_file
    from ovito.vis import Viewport, TachyonRenderer

    renderer = TachyonRenderer()
    renderer.antialiasing = True
    renderer.shadows = False
    renderer.ambient_occlusion = False

    viewport = Viewport()
    viewport.type = Viewport.Type.Perspective

    cached_pipeline = None
    cached_file = None

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue

        try:
            req = json.loads(raw)
        except Exception:
            continue

        if req.get('type') == 'shutdown':
            break

        try:
            dump_file        = req['dump_file']
            ovito_script     = req.get('ovito_script')
            camera_angle     = req.get('camera_angle', 'default')
            render_size      = tuple(req.get('render_size', [600, 450]))
            file_frame_index = req.get('file_frame_index', 0)

            # Update camera angle
            if camera_angle == 'top':
                viewport.camera_pos = (0, 0, 50)
                viewport.camera_dir = (0, 0, -1)
            elif camera_angle == 'side':
                viewport.camera_pos = (50, 0, 0)
                viewport.camera_dir = (-1, 0, 0)
            else:
                viewport.camera_pos = (30, 30, 30)
                viewport.camera_dir = (-1, -1, -1)

            # Import pipeline only when file changes
            if cached_file != dump_file:
                if cached_pipeline is not None:
                    try:
                        cached_pipeline.remove_from_scene()
                    except Exception:
                        pass
                    for p in list(ovito.scene.pipelines):
                        try:
                            p.remove_from_scene()
                        except Exception:
                            pass

                cached_pipeline = import_file(dump_file)
                cached_pipeline.modifiers.clear()

                if ovito_script:
                    try:
                        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                        from ovito_executor import DynamicOVITOExecutor
                        data = cached_pipeline.compute(0)
                        executor = DynamicOVITOExecutor()
                        cached_pipeline = executor.execute_script(ovito_script, cached_pipeline, data)
                    except Exception as e:
                        print(json.dumps({'status': 'warn', 'msg': f'custom script failed: {e}'}),
                              flush=True)
                        apply_basic_viz(cached_pipeline)
                else:
                    apply_basic_viz(cached_pipeline)

                cached_pipeline.add_to_scene()
                try:
                    cell_vis = cached_pipeline.source.data.cell.vis
                    cell_vis.rendering_color = (1.0, 1.0, 1.0)
                    cell_vis.line_width = 0.5
                except Exception:
                    pass

                cached_file = dump_file

            viewport.zoom_all()

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
            try:
                try:
                    viewport.render_image(
                        filename=tmp_path,
                        size=render_size,
                        alpha=True,
                        renderer=renderer,
                        frame=file_frame_index,
                    )
                except Exception as render_err:
                    # Custom script pipeline failed at render time — fall back to basic viz silently
                    sys.stderr.write(f'[RENDERER] falling back to basic viz: {render_err}\n')
                    sys.stderr.flush()
                    cached_pipeline.modifiers.clear()
                    apply_basic_viz(cached_pipeline)
                    viewport.render_image(
                        filename=tmp_path,
                        size=render_size,
                        alpha=True,
                        renderer=renderer,
                        frame=file_frame_index,
                    )

                with open(tmp_path, 'rb') as f:
                    image_b64 = base64.b64encode(f.read()).decode('utf-8')
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            print(json.dumps({'status': 'ok', 'image': image_b64}), flush=True)

        except Exception as e:
            import traceback
            print(json.dumps({'status': 'error', 'error': str(e),
                              'tb': traceback.format_exc()}), flush=True)


if __name__ == '__main__':
    main()
