"""Serve a browser-based viewer for raw SC2 observations."""

import asyncio
import base64
import collections
import importlib.resources as resources
import json
from pathlib import Path
import threading
import time

from absl import logging
import numpy as np
from pysc2.lib import buffs
from pysc2.lib import colors
from pysc2.lib import features
from pysc2.lib import point
from pysc2.lib import static_data as static_data_lib
from pysc2.lib import transform
import websockets
from websockets.protocol import State as WSState

from s2clientprotocol import error_pb2 as sc_err
from s2clientprotocol import raw_pb2 as sc_raw
from s2clientprotocol import sc2api_pb2 as sc_pb

_ASSET_DIR = "web_renderer_assets"
_ASSET_CACHE = {}
_RGB_CAMERA_WIDTH = 24
_ActionMarker = collections.namedtuple(
    "_ActionMarker", ["ability_id", "color", "pos"])


def _read_asset(name):
  cached = _ASSET_CACHE.get(name)
  if cached is not None:
    return cached
  try:
    data = resources.files(__package__).joinpath(_ASSET_DIR, name).read_bytes()
  except Exception:
    data = Path(__file__).with_name(_ASSET_DIR).joinpath(name).read_bytes()
  _ASSET_CACHE[name] = data
  return data


def _enum_name(enum_type, value):
  try:
    return enum_type.Name(value)
  except ValueError:
    return str(value)


def _safe_buff_name(value):
  try:
    return buffs.Buffs(value).name
  except ValueError:
    return str(value)


def _effect_name(value):
  try:
    return features.Effects(value).name
  except ValueError:
    return str(value)


def _rgb_triplet(color):
  arr = np.asarray(color)
  return [int(arr[0]), int(arr[1]), int(arr[2])]


class _WebViewerStateBuilder:
  """Builds a serializable viewer model from raw SC2 observations."""

  def __init__(self, game_info, static_data, *, feature_screen,
               feature_minimap, composite_player_relative):
    self._game_info = game_info
    self._static_data = static_data
    self._screen_features = self._build_feature_map(features.SCREEN_FEATURES)
    self._minimap_features = self._build_feature_map(features.MINIMAP_FEATURES)
    self._default_view = self._build_default_view(
        feature_screen, feature_minimap, composite_player_relative)

    self._map_size = point.Point.build(game_info.start_raw.map_size)
    self._playable = point.Rect(
        point.Point.build(game_info.start_raw.playable_area.p0),
        point.Point.build(game_info.start_raw.playable_area.p1))

    self._world_to_world_tl = transform.Linear(
        point.Point(1, -1), point.Point(0, self._map_size.y))
    self._world_tl_to_world_camera_rel = transform.Linear(
        offset=-self._map_size / 4)

    self._feature_screen_px = None
    self._feature_minimap_px = None
    self._feature_camera_width_world_units = None
    self._world_to_feature_screen = None
    self._world_to_feature_screen_px = None
    self._world_to_feature_minimap = None
    self._world_to_feature_minimap_px = None
    self._feature_camera = None

    self._rgb_screen_px = None
    self._rgb_minimap_px = None
    self._world_to_rgb_screen = None
    self._world_to_rgb_screen_px = None
    self._world_to_rgb_minimap = None
    self._world_to_rgb_minimap_px = None
    self._rgb_camera = None

    if game_info.options.HasField("feature_layer"):
      fl_opts = game_info.options.feature_layer
      self._feature_screen_px = point.Point.build(fl_opts.resolution)
      self._feature_minimap_px = point.Point.build(fl_opts.minimap_resolution)
      self._feature_camera_width_world_units = fl_opts.width

      world_camera_rel_to_feature_screen = transform.Linear(
          self._feature_screen_px / self._feature_camera_width_world_units,
          self._feature_screen_px / 2)
      self._world_to_feature_screen = transform.Chain(
          self._world_to_world_tl,
          self._world_tl_to_world_camera_rel,
          world_camera_rel_to_feature_screen)
      self._world_to_feature_screen_px = transform.Chain(
          self._world_to_feature_screen,
          transform.PixelToCoord())

      world_tl_to_feature_minimap = transform.Linear(
          self._feature_minimap_px / self._playable.diagonal.max_dim())
      world_tl_to_feature_minimap.offset = world_tl_to_feature_minimap.fwd_pt(
          -self._world_to_world_tl.fwd_pt(self._playable.bl))
      self._world_to_feature_minimap = transform.Chain(
          self._world_to_world_tl,
          world_tl_to_feature_minimap)
      self._world_to_feature_minimap_px = transform.Chain(
          self._world_to_feature_minimap,
          transform.PixelToCoord())

    if game_info.options.HasField("render"):
      render_opts = game_info.options.render
      self._rgb_screen_px = point.Point.build(render_opts.resolution)
      self._rgb_minimap_px = point.Point.build(render_opts.minimap_resolution)

      world_camera_rel_to_rgb_screen = transform.Linear(
          self._rgb_screen_px / _RGB_CAMERA_WIDTH,
          self._rgb_screen_px / 2)
      self._world_to_rgb_screen = transform.Chain(
          self._world_to_world_tl,
          self._world_tl_to_world_camera_rel,
          world_camera_rel_to_rgb_screen)
      self._world_to_rgb_screen_px = transform.Chain(
          self._world_to_rgb_screen,
          transform.PixelToCoord())

      world_tl_to_rgb_minimap = transform.Linear(
          self._rgb_minimap_px / self._map_size.max_dim())
      self._world_to_rgb_minimap = transform.Chain(
          self._world_to_world_tl,
          world_tl_to_rgb_minimap)
      self._world_to_rgb_minimap_px = transform.Chain(
          self._world_to_rgb_minimap,
          transform.PixelToCoord())

  @staticmethod
  def _build_feature_map(feature_set):
    return {name: feature for name, feature in feature_set._asdict().items()}

  @staticmethod
  def _mode_for(value):
    if value in ("composite", "base", "base_map"):
      return "composite"
    return "layer"

  def _build_default_view(self, screen, minimap, overlay_pr):
    del overlay_pr
    return {
        "screen_mode": self._mode_for(screen),
        "minimap_mode": self._mode_for(minimap),
        "screen_layer": screen if self._mode_for(screen) == "layer" else "height_map",
        "minimap_layer": minimap if self._mode_for(minimap) == "layer" else "height_map",
        "effects": True,
        "units": True,
        "labels": True,
        "camera": True,
    }

  def static_payload(self):
    def enum_labels(enum_cls):
      return [{"value": int(member.value), "label": member.name}
              for member in enum_cls]

    def bool_labels(off_label="off", on_label="on"):
      return [
          {"value": 0, "label": off_label},
          {"value": 1, "label": on_label},
      ]

    def legend_labels(name, feature):
      if name == "player_relative":
        return enum_labels(features.PlayerRelative)
      if name == "visibility_map":
        return enum_labels(features.Visibility)
      if name == "effects":
        return enum_labels(features.Effects)
      if name == "creep":
        return bool_labels("none", "creep")
      if name == "power":
        return bool_labels("none", "power")
      if name == "camera":
        return bool_labels("none", "camera")
      if name == "selected":
        return bool_labels("no", "selected")
      if name == "alerts":
        return bool_labels("no", "alert")
      if name == "pathable":
        return bool_labels("blocked", "pathable")
      if name == "buildable":
        return bool_labels("blocked", "buildable")
      if name == "hallucinations":
        return bool_labels("no", "hallucination")
      if name == "cloaked":
        return bool_labels("visible", "cloaked")
      if name == "blip":
        return bool_labels("no", "blip")
      if name == "active":
        return bool_labels("idle", "active")
      if name == "placeholder":
        return bool_labels("no", "placeholder")
      if feature.type == features.FeatureType.CATEGORICAL and feature.scale <= 8:
        return [{"value": value, "label": str(value)}
                for value in range(int(feature.scale))]
      return []

    def pack_layers(feature_map):
      layers = []
      for name, feature in feature_map.items():
        palette = np.asarray(feature.palette, dtype=np.uint8).reshape(-1)
        layers.append({
            "name": name,
            "scale": int(feature.scale),
            "type": feature.type.name.lower(),
            "palette": palette.tolist(),
            "legend_labels": legend_labels(name, feature),
        })
      return layers

    return {
        "type": "static",
        "screen_layers": pack_layers(self._screen_features),
        "minimap_layers": pack_layers(self._minimap_features),
        "default_view": self._default_view,
    }

  def _resolve_mode(self, requested_mode, obs):
    has_rgb = (
        obs.observation.HasField("render_data") and
        obs.observation.render_data.HasField("map") and
        obs.observation.render_data.HasField("minimap") and
        self._rgb_screen_px is not None)
    has_feature = self._feature_screen_px is not None
    if requested_mode == "rgb":
      return "rgb" if has_rgb else "feature"
    if requested_mode == "feature":
      return "feature" if has_feature else "rgb"
    if has_rgb:
      return "rgb"
    if has_feature:
      return "feature"
    return None

  def _update_camera(self, camera_center):
    self._world_tl_to_world_camera_rel.offset = (
        -self._world_to_world_tl.fwd_pt(camera_center) *
        self._world_tl_to_world_camera_rel.scale)

    if self._feature_screen_px is not None:
      self._feature_camera = self._camera_rect(
          camera_center,
          self._feature_screen_px,
          self._feature_camera_width_world_units)
    if self._rgb_screen_px is not None:
      self._rgb_camera = self._camera_rect(
          camera_center,
          self._rgb_screen_px,
          _RGB_CAMERA_WIDTH)

  def _camera_rect(self, camera_center, screen_px, width_world_units):
    camera_radius = (
        screen_px / screen_px.x * width_world_units / 2)
    center = camera_center.bound(camera_radius, self._map_size - camera_radius)
    return point.Rect(
        (center - camera_radius).bound(self._map_size),
        (center + camera_radius).bound(self._map_size))

  def _screen_transform(self, mode):
    if mode == "rgb":
      return self._world_to_rgb_screen, self._rgb_screen_px
    return self._world_to_feature_screen, self._feature_screen_px

  def _minimap_transform(self, mode):
    if mode == "rgb":
      return self._world_to_rgb_minimap, self._rgb_minimap_px
    return self._world_to_feature_minimap, self._feature_minimap_px

  def _camera_for_mode(self, mode):
    if mode == "rgb":
      return self._rgb_camera
    return self._feature_camera

  def _screen_point_for_mode(self, mode, world_pos):
    world_to_screen, screen_size = self._screen_transform(mode)
    if world_to_screen is None or screen_size is None:
      return None
    screen_pos = world_to_screen.fwd_pt(world_pos)
    if (screen_pos.x < 0 or screen_pos.y < 0 or
        screen_pos.x >= screen_size.x or screen_pos.y >= screen_size.y):
      return None
    return screen_pos

  def _screen_anchor_for_mode(self, mode, world_pos):
    screen_pos = self._screen_point_for_mode(mode, world_pos)
    if screen_pos is None:
      return None
    return screen_pos.round()

  def _minimap_point_for_mode(self, mode, world_pos):
    world_to_minimap, minimap_size = self._minimap_transform(mode)
    if world_to_minimap is None or minimap_size is None:
      return None
    minimap_pos = world_to_minimap.fwd_pt(world_pos)
    if (minimap_pos.x < 0 or minimap_pos.y < 0 or
        minimap_pos.x >= minimap_size.x or minimap_pos.y >= minimap_size.y):
      return None
    return minimap_pos

  def _minimap_anchor_for_mode(self, mode, world_pos):
    minimap_pos = self._minimap_point_for_mode(mode, world_pos)
    if minimap_pos is None:
      return None
    return minimap_pos.round()

  @staticmethod
  def _layer_dtype(scale):
    if scale <= 256:
      return np.uint8, "u8"
    if scale <= 65535:
      return np.uint16, "u16"
    return np.uint32, "u32"

  def _encode_layer(self, layer, feature):
    arr = np.asarray(layer)
    h, w = arr.shape
    dtype, dtype_name = self._layer_dtype(int(feature.scale))
    if arr.dtype != dtype:
      arr = arr.astype(dtype, copy=False)
    data = base64.b64encode(arr.tobytes()).decode("ascii")
    return {
        "name": feature.name,
        "w": int(w),
        "h": int(h),
        "dtype": dtype_name,
        "data": data,
    }

  def _encode_layers_from_obs(self, obs, names, feature_map):
    out = {}
    for name in names:
      feature = feature_map.get(name)
      if feature is None:
        continue
      layer = feature.unpack(obs.observation)
      if layer is None:
        continue
      out[name] = self._encode_layer(layer, feature)
    return out

  @staticmethod
  def _encode_frame(frame):
    arr = np.asarray(frame, dtype=np.uint8)
    h, w, _ = arr.shape
    data = base64.b64encode(arr.tobytes()).decode("ascii")
    return {"w": int(w), "h": int(h), "data": data}

  def _build_camera_overlay(self, mode):
    camera_rect = self._camera_for_mode(mode)
    world_to_minimap, _ = self._minimap_transform(mode)
    if camera_rect is None or world_to_minimap is None:
      return None
    tl = world_to_minimap.fwd_pt(camera_rect.tl)
    br = world_to_minimap.fwd_pt(camera_rect.br)
    return [float(tl.x), float(tl.y), float(br.x - tl.x), float(br.y - tl.y)]

  def _build_action_markers(self, mode, action_markers):
    screen_markers = []
    minimap_markers = []
    for act in action_markers:
      if not act.pos or act.color is None:
        continue
      color = _rgb_triplet(act.color)
      if isinstance(act.pos, point.Point):
        screen_pos = self._screen_anchor_for_mode(mode, act.pos)
        minimap_pos = self._minimap_anchor_for_mode(mode, act.pos)
        if screen_pos is not None:
          screen_markers.append({
              "kind": "circle",
              "x": float(screen_pos.x),
              "y": float(screen_pos.y),
              "radius": 2.4,
              "color": color,
          })
        if minimap_pos is not None:
          minimap_markers.append({
              "kind": "circle",
              "x": float(minimap_pos.x),
              "y": float(minimap_pos.y),
              "radius": 1.8,
              "color": color,
          })
      elif isinstance(act.pos, point.Rect):
        screen_tl = self._screen_point_for_mode(mode, act.pos.tl)
        screen_br = self._screen_point_for_mode(mode, act.pos.br)
        minimap_tl = self._minimap_point_for_mode(mode, act.pos.tl)
        minimap_br = self._minimap_point_for_mode(mode, act.pos.br)
        if screen_tl is not None and screen_br is not None:
          screen_markers.append({
              "kind": "rect",
              "x": float(screen_tl.x),
              "y": float(screen_tl.y),
              "w": float(screen_br.x - screen_tl.x),
              "h": float(screen_br.y - screen_tl.y),
              "color": color,
          })
        if minimap_tl is not None and minimap_br is not None:
          minimap_markers.append({
              "kind": "rect",
              "x": float(minimap_tl.x),
              "y": float(minimap_tl.y),
              "w": float(minimap_br.x - minimap_tl.x),
              "h": float(minimap_br.y - minimap_tl.y),
              "color": color,
          })
    return {"screen": screen_markers, "minimap": minimap_markers}

  def _visible_units(self, obs):
    return sorted(
        obs.observation.raw_data.units,
        key=lambda u: (u.pos.z, u.owner != 16, -u.radius, u.tag))

  def _unit_color(self, owner):
    index = owner if owner < len(colors.PLAYER_ABSOLUTE_PALETTE) else 0
    color = colors.PLAYER_ABSOLUTE_PALETTE[index]
    return [int(color[0]), int(color[1]), int(color[2])]

  def _build_units_overlay(self, obs, mode):
    world_to_screen, screen_size = self._screen_transform(mode)
    if world_to_screen is None or screen_size is None:
      return []

    out = []
    for unit in self._visible_units(obs):
      world_pos = point.Point.build(unit.pos)
      screen_pos = self._screen_anchor_for_mode(mode, world_pos)
      if screen_pos is None:
        continue
      if unit.display_type == sc_raw.Hidden:
        continue
      radius = max(1.0, float(int(world_to_screen.fwd_dist(unit.radius))))
      unit_name = self._static_data.units.get(unit.unit_type, str(unit.unit_type))
      detail = ""
      if unit.ideal_harvesters > 0:
        detail = f"{unit.assigned_harvesters} / {unit.ideal_harvesters}"
      elif unit.mineral_contents > 0:
        detail = str(unit.mineral_contents)
      elif unit.vespene_contents > 0:
        detail = str(unit.vespene_contents)
      elif unit.display_type == sc_raw.Snapshot:
        detail = "snapshot"
      elif unit.display_type == sc_raw.Placeholder:
        detail = "placeholder"
      elif unit.is_hallucination:
        detail = "hallucination"
      elif unit.is_burrowed:
        detail = "burrowed"
      elif unit.cloak != sc_raw.NotCloaked:
        detail = "cloaked"
      out.append({
          "x": float(screen_pos.x),
          "y": float(screen_pos.y),
          "radius": radius,
          "color": self._unit_color(unit.owner),
          "name": unit_name,
          "detail": detail,
          "selected": bool(unit.is_selected),
          "health_ratio": float(unit.health / unit.health_max) if unit.health_max else 0.0,
          "shield_ratio": float(unit.shield / unit.shield_max) if unit.shield_max else 0.0,
          "energy_ratio": float(unit.energy / unit.energy_max) if unit.energy_max else 0.0,
      })
    return out

  def _build_unit_paths(self, obs, mode):
    world_to_screen, screen_size = self._screen_transform(mode)
    if world_to_screen is None or screen_size is None:
      return []

    player_id = obs.observation.player_common.player_id
    unit_dict = {unit.tag: unit for unit in obs.observation.raw_data.units}
    paths = []
    for unit in self._visible_units(obs):
      if unit.display_type == sc_raw.Hidden:
        continue
      if player_id not in (0, 16) and unit.owner != player_id:
        continue
      if not unit.orders and not unit.rally_targets:
        continue

      start_world = point.Point.build(unit.pos)
      start_screen = self._screen_anchor_for_mode(mode, start_world)
      if start_screen is None:
        continue

      target_world = None
      if unit.orders:
        order = unit.orders[0]
        if order.HasField("target_world_space_pos"):
          target_world = point.Point.build(order.target_world_space_pos)
        elif order.HasField("target_unit_tag"):
          target_unit = unit_dict.get(order.target_unit_tag)
          if target_unit:
            target_world = point.Point.build(target_unit.pos)
      elif unit.rally_targets:
        target_world = point.Point.build(unit.rally_targets[0].point)

      if target_world is None:
        continue

      target_screen = self._screen_anchor_for_mode(mode, target_world)
      if target_screen is not None:
        paths.append({
            "unit_tag": int(unit.tag),
            "color": _rgb_triplet(colors.cyan),
            "segments": [{
                "x1": float(start_screen.x),
                "y1": float(start_screen.y),
                "x2": float(target_screen.x),
                "y2": float(target_screen.y),
            }],
        })
    return paths

  def _build_effects_overlay(self, obs, mode):
    world_to_screen, screen_size = self._screen_transform(mode)
    if world_to_screen is None or screen_size is None:
      return []

    out = []
    for effect in obs.observation.raw_data.effects:
      if effect.effect_id < len(colors.effects):
        base_color = colors.effects[effect.effect_id]
      else:
        base_color = colors.white
      if not isinstance(base_color, np.ndarray):
        base_color = np.asarray(base_color, dtype=np.uint8)
      color = [int(base_color[0]), int(base_color[1]), int(base_color[2])]
      name = _effect_name(effect.effect_id)
      for pos in effect.pos:
        world_pos = point.Point.build(pos)
        screen_pos = self._screen_anchor_for_mode(mode, world_pos)
        if screen_pos is None:
          continue
        radius = max(1.0, float(int(world_to_screen.fwd_dist(effect.radius))))
        out.append({
            "x": float(screen_pos.x),
            "y": float(screen_pos.y),
            "radius": radius,
            "color": color,
            "name": name,
        })
    return out

  def _build_hud(self, obs, stats):
    player = obs.observation.player_common
    score = obs.observation.score.score
    game_loop = int(obs.observation.game_loop)
    seconds = int(game_loop // 22.4)
    details = obs.observation.score.score_details
    return {
        "minerals": int(player.minerals),
        "vespene": int(player.vespene),
        "food_used": int(player.food_used),
        "food_cap": int(player.food_cap),
        "score": int(score),
        "step": game_loop,
        "time": f"{seconds // 60}:{seconds % 60:02d}",
        "game_rate": round(stats.get("game_rate", 0.0), 1),
        "observed_fps": round(stats.get("observed_fps", 0.0), 1),
        "render_fps": round(stats.get("render_fps", 0.0), 1),
        "apm": int(details.current_apm),
        "epm": int(details.current_effective_apm),
    }

  def _build_selection_panel(self, obs):
    def unit_name(unit_type):
      return self._static_data.units.get(unit_type, "<unknown>")

    def format_progress(value):
      return f"{int(value * 100)}%"

    sections = []
    ui = obs.observation.ui_data

    if ui.groups:
      sections.append({
          "title": "Control Groups",
          "entries": [
              f"{group.control_group_index}: {group.count} {unit_name(group.leader_unit_type)}"
              for group in ui.groups
          ],
      })

    if ui.HasField("single"):
      entries = [
          unit_name(ui.single.unit.unit_type),
          f"Health: {ui.single.unit.health} / {ui.single.unit.max_health}",
      ]
      if ui.single.unit.max_shields:
        entries.append(
            f"Shields: {ui.single.unit.shields} / {ui.single.unit.max_shields}")
      if ui.single.unit.max_energy:
        entries.append(
            f"Energy: {ui.single.unit.energy} / {ui.single.unit.max_energy}")
      if ui.single.unit.build_progress > 0:
        entries.append(
            f"Progress: {format_progress(ui.single.unit.build_progress)}")
      if ui.single.unit.transport_slots_taken > 0:
        entries.append(f"Slots: {ui.single.unit.transport_slots_taken}")
      if ui.single.attack_upgrade_level:
        entries.append(f"Attack upgrade: {ui.single.attack_upgrade_level}")
      if ui.single.armor_upgrade_level:
        entries.append(f"Armor upgrade: {ui.single.armor_upgrade_level}")
      if ui.single.shield_upgrade_level:
        entries.append(f"Shield upgrade: {ui.single.shield_upgrade_level}")
      for buff_id in ui.single.buffs:
        entries.append(f"Buff: {_safe_buff_name(buff_id)}")
      sections.append({"title": "Selection", "entries": entries})
    elif ui.HasField("multi"):
      counts = collections.defaultdict(int)
      for unit in ui.multi.units:
        counts[unit_name(unit.unit_type)] += 1
      sections.append({
          "title": "Selection",
          "entries": [f"{count} {name}" for name, count in sorted(counts.items())],
      })
    elif ui.HasField("cargo"):
      entries = [
          unit_name(ui.cargo.unit.unit_type),
          f"Empty slots: {ui.cargo.slots_available}",
      ]
      counts = collections.defaultdict(int)
      for unit in ui.cargo.passengers:
        counts[unit_name(unit.unit_type)] += 1
      entries.extend(f"{count} {name}" for name, count in sorted(counts.items()))
      sections.append({"title": "Cargo", "entries": entries})
    elif ui.HasField("production"):
      entries = [unit_name(ui.production.unit.unit_type)]
      if ui.production.production_queue:
        for item in ui.production.production_queue:
          specific = self._static_data.abilities[item.ability_id]
          general = specific
          if specific.remaps_to_ability_id:
            general = self._static_data.abilities.get(
                specific.remaps_to_ability_id, specific)
          name = (general.friendly_name or general.button_name or general.link_name)
          if item.build_progress > 0:
            name += f": {format_progress(item.build_progress)}"
          entries.append(name)
      elif ui.production.build_queue:
        for item in ui.production.build_queue:
          name = unit_name(item.unit_type)
          if item.build_progress > 0:
            name += f": {format_progress(item.build_progress)}"
          entries.append(name)
      sections.append({"title": "Production", "entries": entries})

    upgrades = [
        self._static_data.upgrades[upgrade_id].name
        for upgrade_id in obs.observation.raw_data.player.upgrade_ids
        if upgrade_id in self._static_data.upgrades
    ]
    if upgrades:
      sections.append({"title": "Upgrades", "entries": sorted(upgrades)})

    return sections

  def build_frame(self, obs, *, requested_mode, screen_layers, minimap_layers,
                  alerts, stats, action_markers, screen_px, minimap_px):
    mode = self._resolve_mode(requested_mode, obs)
    if mode is None:
      return None

    camera_center = point.Point.build(obs.observation.raw_data.player.camera)
    self._update_camera(camera_center)

    payload = {
        "type": "frame",
        "mode": mode,
        "screen_px": screen_px,
        "minimap_px": minimap_px,
        "units": self._build_units_overlay(obs, mode),
        "effects": self._build_effects_overlay(obs, mode),
        "camera": self._build_camera_overlay(mode),
        "hud": self._build_hud(obs, stats),
        "alerts": alerts,
        "selection_panel": self._build_selection_panel(obs),
        "unit_paths": self._build_unit_paths(obs, mode),
        "action_markers": self._build_action_markers(mode, action_markers),
    }

    if mode == "rgb":
      payload["screen"] = self._encode_frame(
          features.Feature.unpack_rgb_image(obs.observation.render_data.map))
      payload["minimap"] = self._encode_frame(
          features.Feature.unpack_rgb_image(obs.observation.render_data.minimap))
      return payload

    payload["screen_layers"] = self._encode_layers_from_obs(
        obs, screen_layers, self._screen_features)
    payload["minimap_layers"] = self._encode_layers_from_obs(
        obs, minimap_layers, self._minimap_features)
    return payload


class WebRenderer:
  """Serve a websocket that streams viewer state to a browser UI."""

  def __init__(self,
               host="0.0.0.0",
               port=8000,
               fps=15,
               mode="feature",
               feature_screen="composite",
               feature_minimap="composite",
               composite_player_relative=True,
               overlay_effects=True,
               overlay_units=True,
               overlay_labels=True,
               overlay_camera=True,
               screen_px=None,
               minimap_px=None):
    self._host = host
    self._port = port
    self._fps = fps
    self._mode = mode
    self._overlay_effects = overlay_effects
    self._overlay_units = overlay_units
    self._overlay_labels = overlay_labels
    self._overlay_camera = overlay_camera
    self._screen_px = self._normalize_px(screen_px)
    self._minimap_px = self._normalize_px(minimap_px)
    self._builder = None
    self._feature_screen = feature_screen
    self._feature_minimap = feature_minimap
    self._composite_player_relative = composite_player_relative

    self._default_config = {
        "screen_layers": [],
        "minimap_layers": [],
        "effects": bool(self._overlay_effects),
        "units": bool(self._overlay_units),
        "labels": bool(self._overlay_labels),
        "camera": bool(self._overlay_camera),
    }
    self._static_payload_json = None

    self._loop = None
    self._thread = None
    self._server = None
    self._send_task = None
    self._clients = set()
    self._client_configs = {}
    self._latest = None
    self._latest_id = 0
    self._frame_count = 0
    self._alerts = {}
    self._latest_action_markers = []
    self._game_times = collections.deque(maxlen=100)
    self._render_times = collections.deque(maxlen=100)
    self._last_time = None
    self._last_obs = None
    self._lock = threading.Lock()
    self._control_lock = threading.Lock()
    self._running = True
    self._single_steps = 0
    self._ready = threading.Event()
    self._stop_event = threading.Event()

  def init(self, game_info, static_data):
    if not isinstance(static_data, static_data_lib.StaticData):
      static_data = static_data_lib.StaticData(static_data)
    self._builder = _WebViewerStateBuilder(
        game_info,
        static_data,
        feature_screen=self._feature_screen,
        feature_minimap=self._feature_minimap,
        composite_player_relative=self._composite_player_relative)
    self._static_payload_json = json.dumps(self._builder.static_payload())
    self._default_config = self._build_default_config(self._builder.static_payload()["default_view"])
    self.start()
    if self._loop:
      asyncio.run_coroutine_threadsafe(self._broadcast_static(), self._loop)

  def start(self):
    if self._thread:
      return
    self._stop_event.clear()
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()
    self._ready.wait(timeout=5)
    logging.info("Web viewer mode=%s fps=%s", self._mode, self._fps)

  def stop(self):
    if not self._loop:
      return
    self._stop_event.set()
    try:
      future = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
      future.result(timeout=3)
    except Exception:
      pass
    if self._thread:
      self._thread.join(timeout=3)
    self._thread = None
    self._loop = None

  close = stop

  def render(self, obs):
    if self._builder is None:
      return

    now = time.time()
    self._prepare_actions(obs)
    if self._last_obs is not None and self._last_time is not None:
      step_delta = max(
          1, obs.observation.game_loop - self._last_obs.observation.game_loop)
      self._game_times.append((now - self._last_time, step_delta))
    self._last_time = now
    self._last_obs = obs

    for alert in obs.observation.alerts:
      self._alerts[_enum_name(sc_pb.Alert, alert)] = now
    for err in obs.action_errors:
      if err.result != sc_err.Success:
        self._alerts[_enum_name(sc_err.ActionResult, err.result)] = now

    stats = {
        "game_rate": self._average(self._game_times, value_index=1),
        "observed_fps": self._average_count(self._game_times),
        "render_fps": self._average_count(self._render_times),
    }

    configs = self._snapshot_configs()
    screen_layers = set()
    minimap_layers = set()
    for cfg in configs:
      screen_layers.update(cfg["screen_layers"])
      minimap_layers.update(cfg["minimap_layers"])

    frame_start = time.time()
    payload = self._builder.build_frame(
        obs,
        requested_mode=self._mode,
        screen_layers=screen_layers,
        minimap_layers=minimap_layers,
        alerts=self._active_alerts(now),
        stats=stats,
        action_markers=self._latest_action_markers,
        screen_px=self._screen_px,
        minimap_px=self._minimap_px)
    if not payload:
      return
    self._render_times.append(time.time() - frame_start)

    self._frame_count += 1
    payload["frame_id"] = self._frame_count
    payload["game_loop"] = int(obs.observation.game_loop)
    payload["control_state"] = self._control_state()
    with self._lock:
      self._latest = payload
      self._latest_id += 1

  def get_actions(self, run_config, controller):
    del run_config, controller
    while not self._stop_event.is_set():
      with self._control_lock:
        if self._running:
          return "step"
        if self._single_steps > 0:
          self._single_steps -= 1
          return "step"
      time.sleep(0.05)
    return "step"

  @staticmethod
  def _average(entries, value_index):
    if not entries:
      return 0.0
    times = sum(item[0] for item in entries) or 1.0
    values = sum(item[value_index] for item in entries)
    return values / times

  @staticmethod
  def _average_count(entries):
    if not entries:
      return 0.0
    first = entries[0]
    if isinstance(first, tuple):
      times = sum(item[0] for item in entries) or 1.0
    else:
      times = sum(entries) or 1.0
    return len(entries) / times

  def _active_alerts(self, now):
    return [
        name for name, ts in sorted(self._alerts.items(), key=lambda item: item[1])
        if now < ts + 3
    ]

  def _prepare_actions(self, obs):
    markers = []

    def add_act(ability_id, color, pos):
      if ability_id and ability_id in self._builder._static_data.abilities:
        ability = self._builder._static_data.abilities[ability_id]
        if ability.remaps_to_ability_id:
          ability_id = ability.remaps_to_ability_id
      markers.append(_ActionMarker(ability_id, color, pos))

    for act in obs.actions:
      if (act.HasField("action_raw") and
          act.action_raw.HasField("unit_command") and
          act.action_raw.unit_command.HasField("target_world_space_pos")):
        pos = point.Point.build(act.action_raw.unit_command.target_world_space_pos)
        add_act(act.action_raw.unit_command.ability_id, colors.yellow, pos)

      if act.HasField("action_feature_layer"):
        act_fl = act.action_feature_layer
        if act_fl.HasField("unit_command"):
          if act_fl.unit_command.HasField("target_screen_coord"):
            pos = self._builder._world_to_feature_screen_px.back_pt(
                point.Point.build(act_fl.unit_command.target_screen_coord))
            add_act(act_fl.unit_command.ability_id, colors.cyan, pos)
          elif act_fl.unit_command.HasField("target_minimap_coord"):
            pos = self._builder._world_to_feature_minimap_px.back_pt(
                point.Point.build(act_fl.unit_command.target_minimap_coord))
            add_act(act_fl.unit_command.ability_id, colors.cyan, pos)
          else:
            add_act(act_fl.unit_command.ability_id, None, None)
        if (act_fl.HasField("unit_selection_point") and
            act_fl.unit_selection_point.HasField("selection_screen_coord")):
          pos = self._builder._world_to_feature_screen_px.back_pt(
              point.Point.build(act_fl.unit_selection_point.selection_screen_coord))
          add_act(None, colors.cyan, pos)
        if act_fl.HasField("unit_selection_rect"):
          for rect in act_fl.unit_selection_rect.selection_screen_coord:
            pos = point.Rect(
                self._builder._world_to_feature_screen_px.back_pt(
                    point.Point.build(rect.p0)),
                self._builder._world_to_feature_screen_px.back_pt(
                    point.Point.build(rect.p1)))
            add_act(None, colors.cyan, pos)

      if act.HasField("action_render"):
        act_rgb = act.action_render
        if act_rgb.HasField("unit_command"):
          if act_rgb.unit_command.HasField("target_screen_coord"):
            pos = self._builder._world_to_rgb_screen_px.back_pt(
                point.Point.build(act_rgb.unit_command.target_screen_coord))
            add_act(act_rgb.unit_command.ability_id, colors.red, pos)
          elif act_rgb.unit_command.HasField("target_minimap_coord"):
            pos = self._builder._world_to_rgb_minimap_px.back_pt(
                point.Point.build(act_rgb.unit_command.target_minimap_coord))
            add_act(act_rgb.unit_command.ability_id, colors.red, pos)
          else:
            add_act(act_rgb.unit_command.ability_id, None, None)
        if (act_rgb.HasField("unit_selection_point") and
            act_rgb.unit_selection_point.HasField("selection_screen_coord")):
          pos = self._builder._world_to_rgb_screen_px.back_pt(
              point.Point.build(act_rgb.unit_selection_point.selection_screen_coord))
          add_act(None, colors.red, pos)
        if act_rgb.HasField("unit_selection_rect"):
          for rect in act_rgb.unit_selection_rect.selection_screen_coord:
            pos = point.Rect(
                self._builder._world_to_rgb_screen_px.back_pt(
                    point.Point.build(rect.p0)),
                self._builder._world_to_rgb_screen_px.back_pt(
                    point.Point.build(rect.p1)))
            add_act(None, colors.red, pos)

    if markers:
      self._latest_action_markers = markers

  @staticmethod
  def _normalize_px(value):
    if not value:
      return None
    try:
      return [int(value.x), int(value.y)]
    except Exception:
      try:
        return [int(value[0]), int(value[1])]
      except Exception:
        return None

  @staticmethod
  def _sanitize_layers(names, default_names):
    return [name for name in names if name in default_names]

  def _build_default_config(self, view):
    screen_layers = []
    minimap_layers = []
    if view["screen_mode"] == "composite":
      screen_layers = ["height_map", "creep", "power", "visibility_map"]
    else:
      screen_layers = [view["screen_layer"]]
    if view["minimap_mode"] == "composite":
      minimap_layers = ["height_map", "creep", "visibility_map"]
    else:
      minimap_layers = [view["minimap_layer"]]
    return {
        "screen_layers": screen_layers,
        "minimap_layers": minimap_layers,
        "effects": bool(view["effects"]),
        "units": bool(view["units"]),
        "labels": bool(view["labels"]),
        "camera": bool(view["camera"]),
    }

  def _snapshot_configs(self):
    with self._lock:
      if self._client_configs:
        return list(self._client_configs.values())
    return [self._default_config]

  def _control_state(self):
    with self._control_lock:
      return {"running": bool(self._running)}

  async def _handle_client_message(self, ws, message):
    try:
      data = json.loads(message)
    except Exception:
      return
    if self._builder is None:
      return
    if data.get("type") == "control":
      command = data.get("command")
      with self._control_lock:
        if command == "start":
          self._running = True
          self._single_steps = 0
        elif command == "stop":
          self._running = False
          self._single_steps = 0
        elif command == "step":
          self._running = False
          self._single_steps = 1
      return
    if data.get("type") != "config":
      return
    static_payload = self._builder.static_payload()
    screen_names = [layer["name"] for layer in static_payload["screen_layers"]]
    minimap_names = [layer["name"] for layer in static_payload["minimap_layers"]]
    with self._lock:
      cfg = self._client_configs.get(ws, dict(self._default_config))
      if "screen_layers" in data:
        cfg["screen_layers"] = self._sanitize_layers(
            data.get("screen_layers") or [], screen_names)
      if "minimap_layers" in data:
        cfg["minimap_layers"] = self._sanitize_layers(
            data.get("minimap_layers") or [], minimap_names)
      if "effects" in data:
        cfg["effects"] = bool(data.get("effects"))
      if "units" in data:
        cfg["units"] = bool(data.get("units"))
      if "labels" in data:
        cfg["labels"] = bool(data.get("labels"))
      if "camera" in data:
        cfg["camera"] = bool(data.get("camera"))
      self._client_configs[ws] = cfg

  def _ws_dead(self, ws):
    if hasattr(ws, "closed"):
      return ws.closed
    if hasattr(ws, "state"):
      return ws.state != WSState.OPEN
    return False

  async def _broadcast_static(self):
    if not self._static_payload_json:
      return
    dead = []
    for ws in list(self._clients):
      if self._ws_dead(ws):
        dead.append(ws)
        continue
      try:
        await ws.send(self._static_payload_json)
      except Exception:
        dead.append(ws)
    for ws in dead:
      self._clients.discard(ws)
      with self._lock:
        self._client_configs.pop(ws, None)

  def _run(self):
    self._loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self._loop)
    self._loop.create_task(self._serve())
    self._loop.run_forever()

  async def _serve(self):
    async def handler(*args):
      if not args:
        return
      ws = args[0]
      path = None
      if len(args) > 1:
        path = args[1]
      else:
        path = getattr(ws, "path", None)
        if path is None:
          req = getattr(ws, "request", None)
          path = getattr(req, "path", None) if req is not None else None
      if path and path != "/ws":
        await ws.close()
        return
      self._clients.add(ws)
      with self._lock:
        self._client_configs[ws] = dict(self._default_config)
      try:
        await ws.send(json.dumps({"type": "hello"}))
        if self._static_payload_json:
          await ws.send(self._static_payload_json)
      except Exception:
        pass
      logging.info("Web viewer client connected (path=%s).", path or "unknown")
      try:
        async for message in ws:
          await self._handle_client_message(ws, message)
      finally:
        self._clients.discard(ws)
        with self._lock:
          self._client_configs.pop(ws, None)
        logging.info("Web viewer client disconnected.")

    try:
      from websockets.http11 import Headers as _WSHeaders
      from websockets.http11 import Response as _WSResponse
    except Exception:
      _WSHeaders = None
      _WSResponse = None

    def _reason(status):
      return {
          200: "OK",
          204: "No Content",
          400: "Bad Request",
          404: "Not Found",
      }.get(status, "OK")

    def _response(status, headers, body):
      if _WSResponse is not None:
        hdrs = _WSHeaders(headers) if _WSHeaders is not None else headers
        return _WSResponse(status, _reason(status), hdrs, body)
      return status, headers, body

    def _get_header(headers, name):
      if headers is None:
        return ""
      try:
        return headers.get(name, "")
      except Exception:
        try:
          return headers[name]
        except Exception:
          return ""

    async def process_request(*args, **kwargs):
      del kwargs
      path = None
      headers = None
      if len(args) == 1:
        req = args[0]
        path = getattr(req, "path", None)
        headers = getattr(req, "headers", None)
        if path is None:
          path = getattr(req, "raw_path", None)
        if headers is None:
          headers = getattr(req, "request_headers", None)
      elif len(args) >= 2:
        req = args[1] if hasattr(args[1], "headers") or hasattr(args[1], "path") else args[0]
        path = getattr(req, "path", None)
        headers = getattr(req, "headers", None)
        if path is None and isinstance(args[0], str):
          path = args[0]
        if headers is None and not isinstance(req, str):
          headers = getattr(req, "request_headers", None)

      if not path:
        path = "/"
      if not isinstance(path, str):
        path = str(path)
      path = path.split("?", 1)[0]
      upgrade = _get_header(headers, "Upgrade").lower()

      if path == "/" or path == "/index.html":
        return _response(200, [("Content-Type", "text/html")], _read_asset("index.html"))
      if path == "/viewer.js":
        return _response(200, [("Content-Type", "text/javascript")], _read_asset("viewer.js"))
      if path == "/favicon.ico":
        return _response(204, [], b"")
      if path == "/ws":
        if upgrade == "websocket":
          return None
        body = b"WebSocket endpoint. Open / in your browser."
        return _response(400, [("Content-Type", "text/plain")], body)
      return _response(404, [("Content-Type", "text/plain")], b"Not found")

    self._server = await websockets.serve(
        handler,
        self._host,
        self._port,
        process_request=process_request,
        max_size=None,
        ping_interval=20,
        ping_timeout=20,
    )
    self._send_task = asyncio.create_task(self._send_loop())
    logging.info("Web viewer listening on http://%s:%s", self._host, self._port)
    self._ready.set()

  async def _send_loop(self):
    last_id = 0
    delay = 1.0 / max(1, self._fps)
    try:
      while not self._stop_event.is_set():
        await asyncio.sleep(delay)
        with self._lock:
          latest = self._latest
          latest_id = self._latest_id
          configs = list(self._client_configs.items())
        if not latest or latest_id == last_id:
          continue
        if not self._clients:
          last_id = latest_id
          continue

        dead = []
        if latest.get("mode") == "rgb":
          for ws, cfg in configs:
            if self._ws_dead(ws):
              dead.append(ws)
              continue
            payload = dict(latest)
            if not cfg["effects"]:
              payload.pop("effects", None)
            if not cfg["units"]:
              payload.pop("units", None)
            if not cfg["camera"]:
              payload.pop("camera", None)
            try:
              await ws.send(json.dumps(payload))
            except Exception:
              dead.append(ws)
        else:
          for ws, cfg in configs:
            if self._ws_dead(ws):
              dead.append(ws)
              continue
            payload = {
                "type": "frame",
                "mode": "feature",
                "frame_id": latest.get("frame_id"),
                "game_loop": latest.get("game_loop"),
                "screen_px": latest.get("screen_px"),
                "minimap_px": latest.get("minimap_px"),
                "hud": latest.get("hud"),
                "alerts": latest.get("alerts"),
                "selection_panel": latest.get("selection_panel"),
                "control_state": latest.get("control_state"),
                "unit_paths": latest.get("unit_paths"),
                "action_markers": latest.get("action_markers"),
                "screen_layers": [
                    latest["screen_layers"][name]
                    for name in cfg["screen_layers"]
                    if name in latest["screen_layers"]
                ],
                "minimap_layers": [
                    latest["minimap_layers"][name]
                    for name in cfg["minimap_layers"]
                    if name in latest["minimap_layers"]
                ],
            }
            if cfg["effects"]:
              payload["effects"] = latest.get("effects")
            if cfg["units"]:
              payload["units"] = latest.get("units")
            if cfg["camera"]:
              payload["camera"] = latest.get("camera")
            try:
              await ws.send(json.dumps(payload))
            except Exception:
              dead.append(ws)
        for ws in dead:
          self._clients.discard(ws)
          with self._lock:
            self._client_configs.pop(ws, None)
        last_id = latest_id
    except Exception as exc:
      logging.error("Web viewer send loop crashed: %s", exc)

  async def _shutdown(self):
    if self._send_task:
      self._send_task.cancel()
      self._send_task = None
    for ws in list(self._clients):
      try:
        await ws.close()
      except Exception:
        pass
    self._clients.clear()
    if self._server:
      self._server.close()
      await self._server.wait_closed()
      self._server = None
    if self._loop:
      self._loop.stop()
