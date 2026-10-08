import hashlib
import json

import numpy as np
import pytest
from typer.testing import CliRunner

from nebius_physical_ai.benchmark import appearance_suites, diagnostic_suites
from nebius_physical_ai.cli import app
from nebius_physical_ai.twin.expert import SceneSampler
from nebius_physical_ai.twin.randomize import MENAGERIE_WRIST, SEEED_WRIST, Randomization, SceneRandomizer
from nebius_physical_ai.twin.scene import SO101_XML, SceneLook, Twin, TwinLayout
from nebius_physical_ai.twin.textures import HELD_OUT_SEED, KINDS, SIZE, texture

LAYOUT = TwinLayout()
needs_model = pytest.mark.skipif(not SO101_XML.exists(), reason="SO-101 menagerie model not fetched")


def looks(n: int, **ranges: object) -> list[SceneLook]:
    randomizer = SceneRandomizer(LAYOUT, seed=5, ranges=Randomization(**ranges))
    return [randomizer.look() for _ in range(n)]


class TestTextures:
    @pytest.mark.parametrize("kind", KINDS)
    def test_a_name_always_gives_the_same_image_and_seeds_differ(self, kind: str) -> None:
        a, b = texture(f"{kind}:1"), texture(f"{kind}:2")
        assert len(a) == SIZE * SIZE * 3 and a == texture(f"{kind}:1") and a != b

    def test_an_unknown_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown texture kind"):
            texture("marble:1")


class TestRandomizedLooks:
    def test_without_visual_options_a_seed_draws_the_scenes_and_looks_it_always_did(self) -> None:
        randomizer, sampler = SceneRandomizer(LAYOUT, seed=5, role_agnostic=True), SceneSampler(LAYOUT, seed=5)
        rows = []
        for _ in range(20):
            episode, look = randomizer.episode(sampler), randomizer.look()
            camera = [round(v, 3) for v in look.camera["pos"]] if look.camera else None
            rows.append(
                [[o.label, round(float(o.x_mm), 1)] for o in episode.objects] + [camera, round(look.floor_grey, 3)]
            )
            assert look.floor_texture == "" and look.wrist_camera is None
        digest = hashlib.sha1(json.dumps(rows, default=float).encode()).hexdigest()[:12]
        assert digest == "5d1441fe4769"  # pinned before visual randomisation existed, same on macOS and Linux

    def test_training_textures_never_use_the_held_out_seeds(self) -> None:
        names = [look.floor_texture for look in looks(400, texture_share=0.5)]
        textured = [n for n in names if n]
        assert 0.4 < len(textured) / len(names) < 0.6
        assert {n.split(":")[0] for n in textured} == set(KINDS)
        assert max(int(n.split(":")[1]) for n in textured) < HELD_OUT_SEED

    def test_wrist_mounts_are_split_and_jittered_within_range(self) -> None:
        cameras = [look.wrist_camera for look in looks(400, wrist_randomize=True)]
        assert 0.4 < sum(c["mount"] == "seeed" for c in cameras) / len(cameras) < 0.6
        for camera in cameras:
            mount = SEEED_WRIST if camera["mount"] == "seeed" else MENAGERIE_WRIST
            assert np.abs(np.subtract(camera["pos"], mount["pos"])).max() <= 0.006 + 1e-9
            assert abs(camera["fovy"] - mount["fovy"]) <= 3.0 + 1e-9
            assert np.linalg.norm(camera["quat"]) == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.parametrize("flags", [["--texture-share", "0.5"], ["--wrist-randomize"]])
    def test_visual_options_without_randomize_are_a_usage_error(self, tmp_path: object, flags: list[str]) -> None:
        assert CliRunner().invoke(app, ["demos", str(tmp_path), *flags]).exit_code == 2


@needs_model
class TestAppearanceSuites:
    def test_they_reuse_clean_manys_scenes_seen_differently(self) -> None:
        clean = next(s for s in diagnostic_suites(LAYOUT) if s.name == "clean_many")
        textured, seeed = appearance_suites(LAYOUT)
        for suite in (textured, seeed):
            assert [c.objects for c in suite.cases] == [c.objects for c in clean.cases] and suite.n == clean.n
        floors = [c.look.floor_texture for c in textured.cases]
        assert {f.split(":")[0] for f in floors} == set(KINDS)
        assert min(int(f.split(":")[1]) for f in floors) >= HELD_OUT_SEED
        assert all(c.look.wrist_camera == SEEED_WRIST for c in seeed.cases)


@needs_model
class TestInTheTwin:
    def test_a_textured_floor_and_a_moved_wrist_camera_compile_and_change_only_the_pictures(self) -> None:
        objects = SceneSampler(LAYOUT, seed=0).sample()
        with (
            Twin.from_objects(objects, LAYOUT) as plain,
            Twin.from_objects(
                objects, LAYOUT, look=SceneLook(floor_texture="wood:3", wrist_camera=SEEED_WRIST)
            ) as seen,
        ):
            plain.settle(0.5)
            seen.settle(0.5)
            np.testing.assert_allclose(plain.block_pos("blue"), seen.block_pos("blue"), atol=1e-9)
            assert not np.array_equal(plain.render("front_cam", (48, 64)), seen.render("front_cam", (48, 64)))
            mount = seen.model.body("camera_mount").id
            hidden = [seen.model.geom_rgba[g, 3] for g in range(seen.model.ngeom) if seen.model.geom_bodyid[g] == mount]
            assert hidden and max(hidden) == 0.0  # the menagerie bracket would block the Seeed camera
