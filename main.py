#NBVAL_IGNORE_OUTPUT

import sys
sys.path.append("/media/fernanda/data1/doutorado/programas/devito")
import numpy as np
import torch
from examples.seismic import Model, demo_model, AcquisitionGeometry, Receiver
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import TimeAxis, plot_image, plot_velocity
from devito import Function
import matplotlib.pyplot as plt
import scipy.ndimage

# Set up inversion parameters.
param = {'t0': 0.,
         'tn': 2000.,              # Simulation last 1 second (1000 ms)
         'f0': 0.008,              # Source peak frequency is 8Hz (0.008 kHz)
         'nshots': 1,              # Number of shots to create gradient from
         'nreceivers': 250,        # Number of receiver locations per shot
         'shape': (500, 251),     # Number of grid points (nx, nz).
         'spacing': (5, 5),   # Grid spacing in m. The domain size is now 1km by 1km.
         'origin': (0, 0),         # Need origin to define relative source and receiver locations.         
         'so': 4,                  # Space order         
         'nbl': 20,                # nb thickness.
         'num_batches': 1,          # number of batches
         'num_epochs': 10}

def get_true_model():
    ''' Define the test phantom; in this case we are using
    a simple circle so we can easily see what is going on.
    '''
    data_path = "/media/fernanda/data1/doutorado/projetos/MLReynam/deepwave/SEAM_Vp_Elastic_N23900_chop.bin"
    vp = 1e-3 * np.fromfile(data_path, dtype='float32', sep="")
    vp = vp.reshape(param['shape'])
    return Model(space_order=param['so'], vp=vp, origin=param['origin'], shape=param['shape'],
                 dtype=np.float32, spacing=param['spacing'], nbl=param['nbl'])  

def get_smooth_model():
    ''' Define the test phantom; in this case we are using
    a simple circle so we can easily see what is going on.
    '''    
    data_path = "/media/fernanda/data1/doutorado/projetos/MLReynam/deepwave/SEAM_Vp_Elastic_N23900_chop.bin"
    vp = 1e-3 * np.fromfile(data_path, dtype='float32', sep="")
    vp = vp.reshape(param['shape'])
    v0 = np.empty(param['shape'],dtype=np.float32)
    v0 = scipy.ndimage.gaussian_filter(vp, sigma=15)
    return Model(space_order=param['so'], vp=v0, origin=param['origin'], shape=param['shape'],
                 dtype=np.float32, spacing=param['spacing'], nbl=param['nbl'])

def dump_shot_data(shot_id, rec, geometry):
    ''' Dump shot data to disk.
    '''
    file = open('shot_%d.bin'%shot_id, "wb")
    scopy = rec.data.copy(order='C')
    file.write(scopy)

def set_geometry(model, xs, xr):
    
    src_coordinates = np.empty((xs.size, 2))        
    src_coordinates[:, 0] = xs
    src_coordinates[:, 1] = 20.  

    # Initialize receivers for synthetic and imaging data
    rec_coordinates = np.empty((xr.size, 2))
    rec_coordinates[:, 0] = xr
    rec_coordinates[:, 1] = 20.

    # Geometry 
    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates,
                                   param['t0'], param['tn'], src_type='Ricker',
                                   f0=param['f0'])

    return geometry

def generate_shotdata(param, model, vel, xs, xr):
    """ Model data
    """
           
    dt = model.critical_dt  
    time_range = TimeAxis(start=param['t0'], stop=param['tn'], step=dt)    
    nt = time_range.num
    data = torch.zeros(nt, xs.size, xr.size)

    for i in range(xs.size):

        print('Modeling for shot {}'.format(i))

        geometry = set_geometry(model, xs[i], xr)
        
        # Set up solver.
        solver = AcousticWaveSolver(model, geometry, space_order=param['so'])

        # Generate synthetic receiver data 
        rec, _, _ = solver.forward(vp=vel)

        data[:,i,:] = torch.tensor(rec.data, requires_grad=True)
    
    return data
    #dump_shot_data(shot_id, rec, geometry)

def main():
    #create observed dataset
    true_model = get_true_model()    
    
    xs = np.linspace(0, true_model.domain_size[0], num=param['nshots'])
    xr = np.linspace(0, true_model.domain_size[0], num=param['nreceivers'])
            
    vp = np.empty(true_model.grid.shape,dtype=np.float32)    
    receiver_amplitudes_true = generate_shotdata(param, true_model, true_model.vp, xs, xr)
     
    #normalizing the predicted receiver amplitudes
    rcv_amps_true_max, _ = receiver_amplitudes_true.max(dim=0, keepdim=True)
    rcv_amps_true_norm = receiver_amplitudes_true / (rcv_amps_true_max.abs() + 1e-10)
        
    # Create initial guess model
    smooth_model = get_smooth_model()  
    model = torch.tensor(smooth_model.vp.data) * 1000
    model.requires_grad = True    
            
    # Set-up inversion
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([{'params': [model], 'lr': 10}])


    
        
    for epoch in range(param['num_epochs']):
        epoch_loss = 0.0
        for it in range(param['num_batches']):
            print('Batch:', it, 'of: ', param['num_batches'])
            optimizer.zero_grad()

            batch_rcv_amps_true = rcv_amps_true_norm[:,it::param['num_batches']]
            batch_x_s = xs[it::param['num_batches']]            
            #batch_x_r = xr[it::param['num_batches']]
                        
            #Forward pass
            smooth_model.vp.data[:,:] = model.detach().numpy() / 1000
            batch_rcv_amps_pred = generate_shotdata(param, true_model, smooth_model.vp, batch_x_s, xr)            
            batch_rcv_amps_pred_max, _ = batch_rcv_amps_pred.max(dim=0, keepdim=True)
            batch_rcv_amps_pred_norm = batch_rcv_amps_pred / (batch_rcv_amps_pred_max.abs() + 1e-10)
            
            #Backpropagation/Inversion             
            loss = criterion(batch_rcv_amps_pred_norm, batch_rcv_amps_true)            
            epoch_loss += loss.item()
            loss.backward()            
            optimizer.step()
            print(batch_rcv_amps_pred_norm.grad)

        print('Epoch:', epoch, 'Loss: ', epoch_loss)
        if epoch % 10 == 0:
            plot_velocity(smooth_model) 
    
    

if __name__ == '__main__':
    main()
   
    