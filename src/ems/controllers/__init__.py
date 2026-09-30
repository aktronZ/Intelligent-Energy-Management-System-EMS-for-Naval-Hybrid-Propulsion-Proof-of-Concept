"""Controllers for the energy management system.

A controller is any object with a `decide` method returning the battery power in
kW, positive for discharge and negative for charge, and the diesel power in kW.
The dispatch in IntegratedSystem is authoritative: it clips whatever a controller
requests to what is physically feasible, so a controller cannot break the power
balance or the state-of-charge window.
"""
