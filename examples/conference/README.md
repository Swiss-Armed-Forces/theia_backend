# Scenario

A small airplane (similar to a Cessna 172; assumed maximum velocity is 50 m/s)
starts from the airport at Friedrichshafen. Its flight path through the Rheintal,
over the Walensee to Airport Zurich is authorized. However, the plane's transponder
stops answering when the plane is close to Sargans (time ``t_off``).


# Behaviour

## BLUE Reaction

- Send the Combat Air Patrol (CAP) to investigate.
- Recognize that the plane is non-cooperative.
- Shoot it down.

# Implementation

The rogue plane is implemented as two controllers: One BLUE and one RED.
The ground truth is provided by the BLUE controller until time ``t_off``, after which
the RED controller takes over.

One could implement this as an ordinary plane (SuicidePlaneController = LivingController[WaypointController], extended by an effector)
shared by a TimedController(start_time, end_time).

# Data source

## Restricted zone

AIRAC AIP SUP: 008/2025

https://www.skybriefing.com/documents/10156/531923/LS_Sup_A_2025_008_en.pdf/9383f427-aee4-73e5-2f58-f6c49c721bc8?t=1766406342056

Retrieved 2026-07-02.

Corresponding GeoJSON file: Circle tool on https://geojson.io/?map=7.87/46.59039/9.42586#map=2/0/20

# Sources
