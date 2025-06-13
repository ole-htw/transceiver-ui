sudo apt install uhd???


git clone ...

cd host/

mkdir build4.8 && cd build4.8

make -j8

sudo make install

git checkout v4.7.0.0

...


uhd_image_loader --args="type=x300,addr=192.168.10.2,fpga=HG"

sudo ldconfig

ldconfig -p | grep libuhd.so

sudo uhd_images_downloader

uhd_image_loader --args="type=x300,addr=192.168.10.2,fpga=HG"

ping 192.168.10.2

uhd_usrp_probe --args addr=192.168.10.2

uhd_image_loader --args="type=x300,addr=192.168.10.2" --fpga-path="<path_to_images>/usrp_x310_fpga_HG.bit"

bandbreite, abtastrate, buffer


./tx_generator.py --amplitude 20000 --fs 25e6 --f 50e3 sinus_tx.bin

./tx_samples_from_file --args addr=192.168.10.2 --rate 25e6 --freq 5.18e9 --ant TX/RX --gain 20 --repeat --file sinus_tx.bin 


./rx_samples_to_file --args addr=192.168.20.2 --rate 25e6 --ref external --freq 5.18e9 --ant RX1 --duration 0.01 --gain 30 --file sinus_rx.bin 




uhd_fft --args addr=192.168.20.2 -f 5.18e9 -s 25e6 -g 40 -A RX1




./rx_samples_to_file --args addr=192.168.20.2 --rate 200e6 --ref external --freq 5.18e9 --ant RX1 --duration 0.001 --gain 40 --file sinus_rx.bin


cmake .. \
  -DENABLE_PYTHON_API=ON \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DUHD_PYTHON_DIR=/usr/local/lib/python3.11/dist-packages 
  
  
./tx_generator.py --waveform zadoffchu \
                  --fs 200e6 --samples 10000 \
                  zc_20kHz.bin
                  
                  
./tx_generator.py --waveform zadoffchu \
                  --fs 100e6 \
                  --samples 4000 \
                  --amplitude 30000 \
                  zc_50kHz.bin
