 #!/bin/bash                                                                                                                                                                        
  set -e                                                                                                                                                                             
                                                                                                                                                                                     
  docker run --rm -v $(pwd):/workspace \                                                                                                                                             
    nvidia/cuda:12.9.1-devel-ubuntu20.04 \                                                                                                                                           
    bash -c '                                                                                                                                                                        
      set -e                                                                                                                                                                         
      cd /workspace                                                                                                                                                                  
                                                                                                                                                                                     
      # Install dependencies                                                                                                                                                         
      export DEBIAN_FRONTEND=noninteractive                                                                                                                                          
      apt-get update -y                                                                                                                                                              
      apt-get install -y python3 python3-pip git cmake ninja-build ccache gcc-10 g++-10 curl                                                                                         
      update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 110 --slave /usr/bin/g++ g++ /usr/bin/g++-10                                                                    
                                                                                                                                                                                     
      # Upgrade pip and install build deps                                                                                                                                           
      python3 -m pip install --upgrade pip setuptools wheel                                                                                                                          
      python3 -m pip install -r requirements/build.txt                                                                                                                               
      python3 -m pip install -r requirements/pyproject.txt                                                                                                                           
                                                                                                                                                                                     
      # Build wheel                                                                                                                                                                  
      python3 setup.py bdist_wheel --dist-dir=/workspace/dist                                                                                                                        
                                                                                                                                                                                     
      # Rename to manylinux                                                                                                                                                          
      cd /workspace/dist                                                                                                                                                             
      for w in *.whl; do                                                                                                                                                             
          [[ "$w" == *"linux"* ]] && mv "$w" "${w/linux/manylinux_2_31}"                                                                                                             
      done                                                                                                                                                                           
                                                                                                                                                                                     
      ls -lh *.whl                                                                                                                                                                   
    ' 
