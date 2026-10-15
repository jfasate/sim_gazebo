"""Self-contained F1Tenth vehicle dynamics for the Gazebo co-simulation plant.

Clean-room implementation (no external dependency on any other workspace/package):
  - pacejka.py : combined-slip Magic Formula tire (pure-slip + friction ellipse)
  - vehicle.py : double-track (4-wheel) planar model with load transfer + RK4

Knowledge references (concepts only, NOT code):
  Pacejka, "Tire and Vehicle Dynamics"; CommonRoad / f1tenth vehicle models.
"""
