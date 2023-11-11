

def set_gear(g):
   value = [
     {
         "field": "target",
         "signal": "Vehicle.Powertrain.Transmission.SelectedGear",
         "value": g,
         "delay": "0",
      },
   ]
   return value

def set_break(b):
   value = [
      {
         "field": "target",
         "signal": "Vehicle.Teleoperation.Brake",
         "value": b,
         "delay": "0",
      }
   ]
   return value

def set_turn(s):
	value = [
	   {
         "field": "target",
         "signal": "Vehicle.Teleoperation.SteeringAngle",
         "value": "s",
         "delay": "0",
      }
   ]
   return value

def set_torque(t) :
   value = [ 
      {
         "field": "target",
         "signal": "Vehicle.Teleoperation.Torque",
         "value": t,
         "delay": "0",
      }
   ]
   return value


def heartbeat():
	value = [
		{
        	"field": "target",
         "signal": "Vehicle.Teleoperation.IsEnabled",
       	"value": "TRUE",
        	"delay": "0", 
      }
	]
   return value
