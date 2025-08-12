The file consists of two modules named pose-estimation and Exercise. Extract the files and save them in a directory on the local system.

**pose-estimation** 

It is used for tracking the different exercises and estimating the skeletal points.



1\. Create a virtual environment -In Anaconda prompt change into the directory where the files are stored. 

python -m venv pose\_estimation

conda activate Pose\_estimation



2\. Install PyTorch with GPU

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

\# Replace cu118 with your CUDA version (cu117, cu121, etc.)



3\. Install OpenCV \& numpy

pip install opencv-python numpy



4\. Install other dependencies 



5\. Run on Webcam in Real-Time 

run the demo2.py using python demo2.py --checkpoint-path weights/checkpoint\_iter\_370000.pth --video 0

Select exercise type:

1\. Bicep Curl

2\. Push-Up

3\. Squat



6\. Run the unity



**Exercise**



Open the file using unity version 2023.2.20f1

Connect the VR headset to the unity(system) through wifi.

Run the MainScene

Based on the selection, it will navigate to the corresponding exercise.





