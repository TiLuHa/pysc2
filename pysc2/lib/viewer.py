"""Shared viewer orchestration for pygame and web renderers."""

import dataclasses


@dataclasses.dataclass
class WebViewerOptions:
  host: str = "0.0.0.0"
  port: int = 8000
  fps: int = 15
  mode: str = "feature"
  feature_screen: str = "composite"
  feature_minimap: str = "composite"
  composite_player_relative: bool = True
  overlay_effects: bool = True
  overlay_units: bool = True
  overlay_labels: bool = True
  overlay_camera: bool = True
  screen_px: object = None
  minimap_px: object = None


@dataclasses.dataclass
class ViewerOptions:
  mode: str = "none"
  pygame_fps: float = 22.4
  pygame_step_mul: int = 1
  pygame_render_sync: bool = False
  pygame_render_feature_grid: bool = True
  web: WebViewerOptions = dataclasses.field(default_factory=WebViewerOptions)


class Viewer:
  """Small adapter that exposes a common lifecycle for viewer backends."""

  def __init__(self, options=None):
    self._options = options or ViewerOptions()
    self._backend = None
    self._pygame_action_cmd = None

  @property
  def enabled(self):
    return self._options.mode != "none"

  def _ensure_backend(self):
    if self._backend is not None or not self.enabled:
      return

    if self._options.mode == "pygame":
      from pysc2.lib import renderer_human  # Lazy import to avoid pygame unless needed.
      self._backend = renderer_human.RendererHuman(
          fps=self._options.pygame_fps,
          step_mul=self._options.pygame_step_mul,
          render_sync=self._options.pygame_render_sync,
          render_feature_grid=self._options.pygame_render_feature_grid)
      self._pygame_action_cmd = renderer_human.ActionCmd
    elif self._options.mode == "web":
      from pysc2.lib import web_renderer
      web = self._options.web
      self._backend = web_renderer.WebRenderer(
          host=web.host,
          port=web.port,
          fps=web.fps,
          mode=web.mode,
          feature_screen=web.feature_screen,
          feature_minimap=web.feature_minimap,
          composite_player_relative=web.composite_player_relative,
          overlay_effects=web.overlay_effects,
          overlay_units=web.overlay_units,
          overlay_labels=web.overlay_labels,
          overlay_camera=web.overlay_camera,
          screen_px=web.screen_px,
          minimap_px=web.minimap_px)
    else:
      raise ValueError(f"Unknown viewer mode: {self._options.mode}")

  def init(self, game_info, static_data):
    self._ensure_backend()
    if self._backend:
      self._backend.init(game_info, static_data)

  def render(self, observation):
    if self._backend:
      self._backend.render(observation)

  def action_cmd(self, run_config, controller):
    if not self._backend:
      return "step"
    cmd = self._backend.get_actions(run_config, controller)
    if self._pygame_action_cmd is None:
      return cmd or "step"
    if cmd == self._pygame_action_cmd.RESTART:
      return "restart"
    if cmd == self._pygame_action_cmd.QUIT:
      return "quit"
    return "step"

  def close(self):
    if self._backend:
      self._backend.close()
      self._backend = None
