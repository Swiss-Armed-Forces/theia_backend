"""
Root controller of a side that is commanded with actions.
"""

from __future__ import annotations

import datetime
import enum
import queue
import threading
from typing import TYPE_CHECKING

import pydantic

from theia.config import UNKNOWN_ID
from theia.simulation.actions import (
    Action,
    ActionExecutedEvent,
    ActionRejectedEvent,
    LaunchAction,
    MoveStockAction,
    RejectionReason,
    ToggleMonostaticRadarAction,
)
from theia.simulation.controllers.camp_site_controller import CampSiteController
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.living_controller import LivingController
from theia.simulation.controllers.switchable_radar import (
    SwitchableMonostaticRadarController,
)
from theia.terrain import AbstractTerrainModel
from theia.types import (
    Controller,
    Entity,
    Event,
    Party,
    Point,
    ShipmentArrivedEvent,
    ShipmentDepartedEvent,
    ShipmentLostEvent,
    SituationalPicture,
    StockLostEvent,
    name_and_id,
)

if TYPE_CHECKING:
    # Not imported at runtime: scenario_import imports this module.
    from theia.simulation.scenario_import import TransportTime


class ShipmentState(enum.Enum):
    BREAKING_CAMP = "BREAKING_CAMP"
    """The stock has left the source site's stock, but the site stays visible."""
    UNDERWAY = "UNDERWAY"
    """The shipment is in transit and invisible."""
    ARRIVED = "ARRIVED"
    """The stock has been added to the target site."""


class Shipment(pydantic.BaseModel):
    """Stock being moved from one camp site to another."""

    id: int
    source_site_id: int
    target_site_id: int
    count: int
    state: ShipmentState
    departure_time: datetime.datetime
    """When breaking camp is done and the shipment leaves"""
    arrival_time: datetime.datetime


class CampSiteState(pydantic.BaseModel):
    """Live state of a camp site."""

    id: int
    name: str
    point: Point
    target_id: int
    can_launch: bool
    ammo_type: str
    n_ammo: int
    """Rounds in stock at the site (excluding shipments)"""
    is_alive: bool
    seconds_until_ready: float
    """Remaining reload time before the next launch [s]; 0 means ready"""


class CampSitesStatus(pydantic.BaseModel):
    """Live state of all camp sites and shipments of a side."""

    time: datetime.datetime
    sites: list[CampSiteState]
    shipments: list[Shipment]


class RadarState(pydantic.BaseModel):
    """
    Live state of a living monostatic radar; dead radars are not reported.
    """

    sensor_id: int
    """ID to switch the radar with (``ToggleMonostaticRadarAction``)"""
    target_id: int
    """ID of the radar in the ground truth"""
    is_on: bool


def radar_states(controller: Controller) -> list[RadarState]:
    """
    State of the monostatic radars a controller reports, for sides whose
    radars cannot be switched: always on.

    Dead radars are not reported by their controller, so they are missing.
    A radar's target ID is that of the target carrying its receiver.

    Raises
    ------
    KeyError
        If a radar has no target carrying its receiver.
    """
    target_ids = {
        target.receiver.id: target.id
        for target in controller.targets
        if target.receiver is not None
    }
    return [
        RadarState(
            sensor_id=sensor.id,
            target_id=target_ids[sensor.receiver.id],
            is_on=True,
        )
        for sensor in controller.monostatic_sensors
    ]


class _Rejection(Exception):
    """An action cannot be executed in the current state."""

    def __init__(self, reason: RejectionReason, message: str):
        super().__init__(message)
        self.reason = reason
        self.message = message


class ActionController(ControllerGroup):
    """
    Root controller of a side that is commanded with actions.

    It only executes actions; who gives them (a player via the server, or an AI)
    is up to the caller of :meth:`assign_action`.

    Its children are the side's controllers, including the camp sites (counted
    stock) and the switchable radars; launched rounds are added to it, too. It
    adds the shipments between the camp sites. Actions are queued by
    :meth:`assign_action`, which is thread-safe, and executed at the start of
    the next :meth:`update`. All checks happen at execution. An executed
    action is logged as an event (with ``id``, ``time`` and ``party`` filled
    in), followed by an ``ActionExecutedEvent`` or an ``ActionRejectedEvent``.
    """

    def __init__(
        self,
        controllers: list[Controller],
        is_blue: bool,
        terrain: AbstractTerrainModel,
        camp_sites: list[LivingController[CampSiteController]],
        transport_times: list[TransportTime],
        radars: list[LivingController[SwitchableMonostaticRadarController]],
    ):
        """
        Parameters
        ----------
        controllers: list[Controller]
            The side's controllers
        camp_sites: list[LivingController[CampSiteController]]
            The side's camp sites, which are part of ``controllers``
        radars: list[LivingController[SwitchableRadarController]]
            The side's radars, which are part of ``controllers``
        """
        super().__init__(controllers)
        self._is_blue = is_blue
        self._party = Party.from_is_blue(is_blue)
        self._terrain = terrain
        self._sites: dict[int, LivingController[CampSiteController]] = {
            s.child.camp_site.id: s for s in camp_sites
        }
        self._transport_times = list(transport_times)
        self._routes: dict[tuple[int, int], TransportTime] = {
            (r.source_site_id, r.target_site_id): r for r in transport_times
        }
        self._radars: dict[
            int, LivingController[SwitchableMonostaticRadarController]
        ] = {r.child.sensor_id: r for r in radars}
        self._shipments: list[Shipment] = []
        self._next_shipment_id = 0
        self._actions: queue.Queue[Action] = queue.Queue()
        self._lock = threading.RLock()
        """Guards the camp sites and shipments against concurrent reads."""
        self._time: datetime.datetime | None = None
        """Time of the latest update; ``None`` before the first update"""

    # ------------------------------------------------------------------
    # Controller interface
    # ------------------------------------------------------------------

    def assign_action(self, action: Action) -> bool:
        self._actions.put(action)
        return True

    def on_event(self, event: Event):
        alive = [s for s in self._sites.values() if s.is_alive]
        # The camp sites are among the children, which handle their kills.
        super().on_event(event)
        for site in alive:
            if not site.is_alive:
                self._on_site_killed(site, event.time)

    def update(self, situational_picture: SituationalPicture, dt: datetime.timedelta):
        time = situational_picture.time
        with self._lock:
            self._time = time
            self._execute_queued_actions(time)
            self._advance_shipments(time)
        super().update(situational_picture, dt)

    # ------------------------------------------------------------------
    # Read access for the server
    # ------------------------------------------------------------------

    @property
    def transport_times(self) -> list[TransportTime]:
        return list(self._transport_times)

    def get_status(self) -> CampSitesStatus:
        """
        Live state of the camp sites and shipments.

        Its time is that of the latest update, or the simulation's start time
        before the first update.

        Raises
        ------
        TheiaException
            If the controller has not been bound to a simulation yet.
        """
        with self._lock:
            time = (
                self._time
                if self._time is not None
                else self.simulation_context.start_time
            )
            return CampSitesStatus(
                time=time,
                sites=[
                    CampSiteState(
                        id=s.child.camp_site.id,
                        name=s.child.camp_site.name,
                        point=s.child.camp_site.point,
                        target_id=s.child.camp_site.target_id,
                        can_launch=s.child.camp_site.can_launch,
                        ammo_type=s.child.camp_site.ammo.type,
                        n_ammo=s.child.camp_site.n_ammo,
                        is_alive=s.is_alive,
                        seconds_until_ready=s.child.seconds_until_ready(time),
                    )
                    for s in self._sites.values()
                ],
                shipments=[s.model_copy() for s in self._shipments],
            )

    def get_radar_states(self) -> list[RadarState]:
        """
        Live state of the radars the player can switch on and off.

        Dead radars are not reported.
        """
        with self._lock:
            return [
                RadarState(
                    sensor_id=sensor_id,
                    target_id=radar.target_id,
                    is_on=radar.child.is_on,
                )
                for sensor_id, radar in self._radars.items()
                if radar.is_alive
            ]

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _execute_queued_actions(self, time: datetime.datetime):
        while True:
            try:
                action = self._actions.get_nowait()
            except queue.Empty:
                return
            self._execute_action(action, time)

    def _execute_action(self, action: Action, time: datetime.datetime):
        # Own the ID, so that the outcome can refer to the action even before
        # the simulator has seen it. Copy, so that the actor's object stays as
        # it was given.
        action = action.model_copy(
            update={
                "id": self.simulation_context.id_provider.increment(Entity.EVENT),
                "time": time,
                "party": self._party,
            }
        )
        self._broadcast_event(action)
        try:
            if isinstance(action, LaunchAction):
                result = self._execute_launch(action, time)
            elif isinstance(action, MoveStockAction):
                result = self._execute_move_stock(action, time)
            elif isinstance(action, ToggleMonostaticRadarAction):
                result = self._execute_toggle_radar(action)
            else:
                raise _Rejection(
                    RejectionReason.UNKNOWN_ACTION_TYPE,
                    f"unknown action type {type(action).__name__}",
                )
        except _Rejection as e:
            self._broadcast_event(
                ActionRejectedEvent(
                    id=UNKNOWN_ID,
                    time=time,
                    party=self._party,
                    action_id=action.id,
                    reason=e.reason,
                    message=e.message,
                )
            )
        else:
            self._broadcast_event(
                ActionExecutedEvent(
                    id=UNKNOWN_ID,
                    time=time,
                    party=self._party,
                    action_id=action.id,
                    result=result,
                )
            )

    @staticmethod
    def _site_label(site: LivingController[CampSiteController]) -> str:
        return name_and_id(site.child.camp_site.name, site.child.camp_site.id)

    def _get_site(self, site_id: int) -> LivingController[CampSiteController]:
        site = self._sites.get(site_id)
        if site is None:
            raise _Rejection(
                RejectionReason.UNKNOWN_CAMP_SITE, f"unknown camp site {site_id}"
            )
        if not site.is_alive:
            raise _Rejection(
                RejectionReason.CAMP_SITE_DESTROYED,
                f"camp site {self._site_label(site)} is destroyed",
            )
        return site

    def _execute_launch(self, action: LaunchAction, time: datetime.datetime) -> str:
        site = self._get_site(action.site_id)
        name = self._site_label(site)
        if not site.child.camp_site.can_launch:
            raise _Rejection(RejectionReason.CANNOT_LAUNCH, f"{name} cannot launch")
        if site.child.camp_site.n_ammo < 1:
            raise _Rejection(RejectionReason.NO_STOCK, f"{name} has no stock")
        seconds_until_ready = site.child.seconds_until_ready(time)
        if seconds_until_ready > 0:
            raise _Rejection(
                RejectionReason.RELOADING,
                f"{name} is reloading for {seconds_until_ready:.0f} s",
            )

        id_provider = self.simulation_context.id_provider
        target_id = id_provider.increment(Entity.TARGET)
        effector_id = id_provider.increment(Entity.EFFECTOR)
        try:
            controller = site.child.camp_site.ammo.to_controller(
                target_id=target_id,
                effector_id=effector_id,
                p_start=site.child.camp_site.point,
                destination=action.destination,
                t_start=time,
                path_generator=action.path_generator,
                terrain=self._terrain,
                is_blue=self._is_blue,
            )
        except ValueError as e:
            raise _Rejection(
                RejectionReason.NO_FLIGHT_PATH, f"no flight path: {e}"
            ) from e

        self.add_controller(controller)
        site.child.camp_site.n_ammo -= 1
        site.child.time_of_last_launch = time
        d = action.destination
        return (
            f"{name} launched {site.child.camp_site.ammo.type} #{target_id} at "
            f"({d.lat:.5f}, {d.lon:.5f}, {d.alt:.0f} m); {site.child.camp_site.n_ammo} left"
        )

    def _execute_move_stock(
        self, action: MoveStockAction, time: datetime.datetime
    ) -> str:
        source = self._get_site(action.source_site_id)
        target = self._get_site(action.target_site_id)
        route = self._routes.get((action.source_site_id, action.target_site_id))
        if route is None:
            raise _Rejection(
                RejectionReason.NO_ROUTE,
                f"no route from camp site {self._site_label(source)} "
                f"to {self._site_label(target)}",
            )
        if action.count > source.child.camp_site.n_ammo:
            raise _Rejection(
                RejectionReason.INSUFFICIENT_STOCK,
                f"{self._site_label(source)} has only "
                f"{source.child.camp_site.n_ammo} rounds in stock",
            )

        source.child.camp_site.n_ammo -= action.count
        departure_time = time + datetime.timedelta(seconds=route.seconds_until_leave)
        shipment = Shipment(
            id=self._next_shipment_id,
            source_site_id=source.child.camp_site.id,
            target_site_id=target.child.camp_site.id,
            count=action.count,
            state=ShipmentState.BREAKING_CAMP,
            departure_time=departure_time,
            arrival_time=departure_time
            + datetime.timedelta(seconds=route.seconds_underway),
        )
        self._next_shipment_id += 1
        self._shipments.append(shipment)
        return (
            f"shipment #{shipment.id}: {shipment.count} rounds from "
            f"{self._site_label(source)} to {self._site_label(target)}, "
            f"departure {departure_time.isoformat()}, "
            f"arrival {shipment.arrival_time.isoformat()}"
        )

    def _execute_toggle_radar(self, action: ToggleMonostaticRadarAction) -> str:
        living = self._radars.get(action.sensor_id)
        if living is None:
            raise _Rejection(
                RejectionReason.UNKNOWN_RADAR,
                f"unknown monostatic radar {action.sensor_id}",
            )
        if not living.is_alive:
            raise _Rejection(
                RejectionReason.RADAR_DESTROYED,
                f"radar {self._radar_label(living)} is destroyed",
            )
        living.child.is_on = action.new_is_on
        return (
            f"radar {self._radar_label(living)} switched "
            f"{'on' if action.new_is_on else 'off'}"
        )

    @staticmethod
    def _radar_label(
        living: LivingController[SwitchableMonostaticRadarController],
    ) -> str:
        radar = living.child
        return name_and_id(radar.child.name, radar.sensor_id)

    # ------------------------------------------------------------------
    # Shipments and losses
    # ------------------------------------------------------------------

    def _advance_shipments(self, time: datetime.datetime):
        remaining: list[Shipment] = []
        for shipment in self._shipments:
            if (
                shipment.state == ShipmentState.BREAKING_CAMP
                and time >= shipment.departure_time
            ):
                shipment.state = ShipmentState.UNDERWAY
                self._broadcast_event(
                    ShipmentDepartedEvent(**self._shipment_fields(shipment, time))
                )
            if (
                shipment.state == ShipmentState.UNDERWAY
                and time >= shipment.arrival_time
            ):
                shipment.state = ShipmentState.ARRIVED
                target = self._sites[shipment.target_site_id]
                if target.is_alive:
                    target.child.camp_site.n_ammo += shipment.count
                    self._broadcast_event(
                        ShipmentArrivedEvent(
                            **self._shipment_fields(shipment, time),
                            n_ammo=target.child.camp_site.n_ammo,
                        )
                    )
                else:
                    self._broadcast_shipment_lost(
                        shipment, shipment.target_site_id, time
                    )
            if shipment.state != ShipmentState.ARRIVED:
                remaining.append(shipment)
        self._shipments = remaining

    def _on_site_killed(
        self, site: LivingController[CampSiteController], time: datetime.datetime
    ):
        """Clear the stock and the shipments breaking camp of a killed site."""
        with self._lock:
            lost_stock = site.child.camp_site.n_ammo
            site.child.camp_site.n_ammo = 0
            lost_shipments = [
                s
                for s in self._shipments
                if s.source_site_id == site.child.camp_site.id
                and s.state == ShipmentState.BREAKING_CAMP
            ]
            self._shipments = [s for s in self._shipments if s not in lost_shipments]
        if lost_stock > 0:
            self._broadcast_event(
                StockLostEvent(
                    id=UNKNOWN_ID,
                    time=time,
                    party=self._party,
                    site_id=site.child.camp_site.id,
                    site_name=site.child.camp_site.name,
                    count=lost_stock,
                )
            )
        for shipment in lost_shipments:
            self._broadcast_shipment_lost(shipment, site.child.camp_site.id, time)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _broadcast_shipment_lost(
        self, shipment: Shipment, destroyed_site_id: int, time: datetime.datetime
    ):
        self._broadcast_event(
            ShipmentLostEvent(
                **self._shipment_fields(shipment, time),
                destroyed_site_id=destroyed_site_id,
            )
        )

    def _shipment_fields(self, shipment: Shipment, time: datetime.datetime) -> dict:
        """Fields common to all ``ShipmentEvent``s."""
        return dict(
            id=UNKNOWN_ID,
            time=time,
            party=self._party,
            shipment_id=shipment.id,
            source_site_id=shipment.source_site_id,
            source_site_name=self._site_name(shipment.source_site_id),
            target_site_id=shipment.target_site_id,
            target_site_name=self._site_name(shipment.target_site_id),
            count=shipment.count,
        )

    def _site_name(self, site_id: int) -> str:
        return self._sites[site_id].child.camp_site.name

    def _broadcast_event(self, event: Event):
        # Own events take the same path as the children's events.
        self._relais.on_event(event)
