import copy
import datetime
import unittest

import numpy as np

from theia.detection.pcl import PclDetector
from theia.detection.pet import PetDetector
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.geojson_controller import GeoJsonController
from theia.simulation.controllers.living_controller import LivingController
from theia.simulation.controllers.monostatic_radar_controller import (
    MonostaticRadarController,
)
from theia.simulation.controllers.switchable_radar import (
    SwitchableMonostaticRadarController,
)
from theia.simulation.damage_model import UniformDamageModel
from theia.simulation.simulator import NeverCriterion, Simulator
from theia.simulation.theia_logging import NoLogger
from theia.simulation.trackers.tracking import DummyTracker
from theia.terrain import DummyTerrain
from theia.types import (
    Controller,
    IdProvider,
    SimulationContext,
    SituationalPicture,
    TheiaException,
)

T0 = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)


class LeafController(Controller):
    def update(self, situational_picture: SituationalPicture, dt: datetime.timedelta):
        pass

    def on_event(self, event):
        pass


class ControllerBindingTest(unittest.TestCase):
    def setUp(self):
        self.context = SimulationContext(id_provider=IdProvider())

    def test_unbound_controller_raises(self):
        with self.assertRaises(TheiaException):
            _ = LeafController().simulation_context

    def test_bind(self):
        leaf = LeafController()
        leaf.bind(self.context)
        self.assertIs(leaf.simulation_context, self.context)

    def test_controller_group_binds_nested_children(self):
        leaf = LeafController()
        group = ControllerGroup([ControllerGroup([leaf])])
        group.bind(self.context)
        self.assertIs(leaf.simulation_context, self.context)

    def test_controller_group_binds_added_child(self):
        group = ControllerGroup([])
        group.bind(self.context)
        leaf = LeafController()
        group.add_controller(leaf)
        self.assertIs(leaf.simulation_context, self.context)

    def test_child_added_before_binding_is_bound_later(self):
        group = ControllerGroup([])
        leaf = LeafController()
        group.add_controller(leaf)
        with self.assertRaises(TheiaException):
            _ = leaf.simulation_context
        group.bind(self.context)
        self.assertIs(leaf.simulation_context, self.context)

    def test_wrappers_bind_their_child(self):
        wrappers = {
            "LivingController": lambda c: LivingController(child=c, target_id=1),
            "GeoJsonController": lambda c: GeoJsonController(
                child=c, geojson_features={}
            ),
        }
        for name, wrap in wrappers.items():
            with self.subTest(wrapper=name):
                leaf = LeafController()
                wrapper = wrap(leaf)
                wrapper.bind(self.context)
                self.assertIs(wrapper.simulation_context, self.context)
                self.assertIs(leaf.simulation_context, self.context)

    def test_switchable_radar_binds_its_child(self):
        # Bypass validation: only the binding is under test.
        radar = MonostaticRadarController.model_construct()
        switchable = SwitchableMonostaticRadarController.model_construct(
            child=radar, sensor_id=1
        )
        switchable.bind(self.context)
        self.assertIs(switchable.simulation_context, self.context)
        self.assertIs(radar.simulation_context, self.context)

    def test_deep_copy_shares_the_context(self):
        leaf = LeafController()
        leaf.bind(self.context)
        self.assertIs(copy.deepcopy(self.context), self.context)
        self.assertIs(
            leaf.model_copy(deep=True).simulation_context.id_provider,
            self.context.id_provider,
        )

    def test_context_is_not_serialised(self):
        leaf = LeafController()
        leaf.bind(self.context)
        data = leaf.model_dump(mode="json")
        self.assertNotIn("_simulation_context", data)
        self.assertNotIn("simulation_context", data)


class SimulatorBindsControllersTest(unittest.TestCase):
    def test_roots_and_descendants_get_the_simulators_id_provider(self):
        id_provider = IdProvider()
        blue_leaf = LeafController()
        red_leaf = LeafController()
        blue = ControllerGroup([LivingController(child=blue_leaf, target_id=1)])
        red = GeoJsonController(child=ControllerGroup([red_leaf]), geojson_features={})
        terrain = DummyTerrain(has_los=True)
        rng = np.random.Generator(np.random.PCG64(seed=1))
        Simulator(
            pcl_detector=PclDetector(),
            pet_detector=PetDetector(terrain_model=terrain),
            blue_controller=blue,
            red_controller=red,
            blue_tracker=DummyTracker(),
            red_tracker=DummyTracker(),
            start_time=T0,
            time_step=datetime.timedelta(seconds=1),
            min_time_per_step=datetime.timedelta(seconds=0),
            termination_criterion=NeverCriterion(),
            rng=rng,
            listener=NoLogger(),
            terrain_model=terrain,
            damage_model=UniformDamageModel(1.0, rng),
            simulate_clutter=False,
            id_provider=id_provider,
        )
        context = blue_leaf.simulation_context
        self.assertIs(context.id_provider, id_provider)
        for controller in (blue, red, blue_leaf, red_leaf):
            self.assertIs(controller.simulation_context, context)


if __name__ == "__main__":
    unittest.main()
