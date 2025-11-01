
import torch

def tv_3d(L):
    """
    Total variation over a 3D LUT volume.
    L: [B,G,G,G,3]
    """
    dx = L[:,1:,:,:,:] - L[:,:-1,:,:,:]
    dy = L[:,:,1:,:,:] - L[:,:, :-1,:,:]
    dz = L[:,:,:,1:,:] - L[:,:,:, :-1,:]
    return (dx.abs().mean() + dy.abs().mean() + dz.abs().mean())

def l2_residual(delta):
    return (delta**2).mean()
