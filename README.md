##### RIA

#### Comandos para todo:

### Cargar el contenedor desde (cmd):
cd ..\..\..
powershell -ExecutionPolicy Bypass -File start.ps1
powershell -ExecutionPolicy Bypass -File terminal.ps1

### Ejecutar los módulos principales:
ros2 run robobo_ros2 robobo_container --ros-args -p ip:=host.docker.internal
ros2 run robobo_ros2 robobo_container --ros-args -p ip:=host.docker.internal -p modules:="['sim']" -p robot_name:="'1'" -p robot_id:=1
ros2 run robobo_ros2 robobo_container --ros-args -p ip:=host.docker.internal -p modules:="['emotion', 'blob']"


### Actualizar el contenedor:
cd ..\..\..
powershell -ExecutionPolicy Bypass -File build.ps1 --robobo
powershell -ExecutionPolicy Bypass -File build.ps1 --pip
powershell -ExecutionPolicy Bypass -File build.ps1 --all