##### RIA

#### Comandos para todo:
python p1_seguir_entrenar.py --calibrar
python p1_seguir_entrenar.py --comprobar
python p1_seguir_entrenar.py --aleatorio --episodios 5
python p1_seguir_entrenar.py --entrenar --algoritmo sac --pasos 10000
python p1_seguir_entrenar.py --evaluar --episodios 5

### Cargar el contenedor desde (cmd):
cd ..\..\..
powershell -ExecutionPolicy Bypass -File start.ps1
powershell -ExecutionPolicy Bypass -File terminal.ps1

### Ejecutar los módulos principales:
ros2 run robobo_ros2 robobo_container --ros-args -p ip:=host.docker.internal
ros2 run robobo_ros2 robobo_container --ros-args -p ip:=host.docker.internal -p modules:="['blob', 'sim']"


### Actualizar el contenedor:
cd ..\..\..
powershell -ExecutionPolicy Bypass -File build.ps1 --robobo
powershell -ExecutionPolicy Bypass -File build.ps1 --pip
powershell -ExecutionPolicy Bypass -File build.ps1 --all