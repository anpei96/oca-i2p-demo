An example implementation of **ODE-driven cross-attention (OCA)** for image-to-point-cloud registration. 

**model_matr_norgb.py** is the main model file. In the lines 179, we add the OCA layer. OCA layer is a plug-and-play module. Its source code can be viewed in **utils_ode.py**

It has two stage to train MATR+OCA. In the first stage, we train MATR with epoch of 20 and get the pth file. In the next, we upload this pth file and change self.is_enable_ode_att=True in 
**model_matr_norgb.py** with the extra epoch of 5~10.
