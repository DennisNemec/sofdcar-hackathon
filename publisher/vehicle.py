

def set_gear(g):
   value = [
     {
         "field": "target",
         "signal": "Vehicle.Powertrain.Transmission.SelectedGear",
         "value": g,
         "delay": "0",
      },
   ]
   value += heartbeat()
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
   value += heartbeat()
   return value

def set_turn(s):
   value = [
      {
         "field": "target",
         "signal": "Vehicle.Teleoperation.SteeringAngle",
         "value": s,
         "delay": "0",
      }
   ]
   value += heartbeat()
   return value

def set_torque(t, delay):
   value = [ 
      {
         "field": "target",
         "signal": "Vehicle.Teleoperation.Torque",
         "value": str(t),
         "delay": str(delay),
      }
   ]
   value += heartbeat()
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

def main():
   print(set_gear(0.8))

main()