import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import scipy.io as sio



def unit_vector(vector):
    return vector / torch.norm(vector)

def unit_vector_linear(vector):
    return vector / torch.norm(vector, dim=1).view(-1, 1)

def angle_loss(v1, v2, reg=100): # abs(cos(angle)) -> between 0-1
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return (v2_u * v1_u).sum().abs() / reg

def angle_loss_linear(v1, v2, reg=100): # abs(cos(angle)) -> between 0-1
    v1_u = unit_vector_linear(v1)
    v2_u = unit_vector_linear(v2)
    return (v2_u * v1_u).sum().abs() / reg

class MultiDirectionZ_nonlinear(nn.Module):
    def __init__(self,dim_z,vocab_size=1000, **kwargs):
        super(MultiDirectionZ_nonlinear, self).__init__()
        self.dim_z = dim_z
        self.vocab_size = vocab_size
        self.direction_num = int(kwargs["direction_num"])
        self.reg = int(kwargs["angle_loss_reg"])
        self.directions = nn.ModuleList([]) # stores all direction parameters

        for _ in range(self.direction_num):
            self.directions.append(nn.Sequential(nn.Linear(self.dim_z, self.dim_z), nn.ReLU(), nn.Linear(self.dim_z, self.dim_z)))
        self.direction_idx_curr = 0  # currently optimized direction index, changes every iteration 

        self.criterion = nn.MSELoss()

    def transform(self,z,y,step_sizes,**kwargs):
        self.direction_idx_curr = (self.direction_idx_curr + 1) % self.direction_num # direction index update
        if y is not None:
            assert(len(y) == z.shape[0])
        interim = step_sizes * self.directions[self.direction_idx_curr](z)
        z_transformed = z + interim
        z_transformed = z.norm() * z_transformed / z_transformed.norm()
        return(z_transformed)
    
    def transform_test(self,z,y,step_sizes,current_way):
        self.direction_idx_curr = current_way # direction index update
        if y is not None:
            assert(len(y) == z.shape[0])
        interim = step_sizes * self.directions[self.direction_idx_curr](z)
        z_transformed = z + interim
        z_transformed = z.norm() * z_transformed / z_transformed.norm()
        return(z_transformed)

    def compute_loss(self, current, target, batch_start, lossfile, z):
        loss = self.criterion(current,target)
        mse_loss = loss.item()
        for idx in range(len(self.directions)):  # calculate the angle losses between the curr direction and all other directions  
            if idx == self.direction_idx_curr:
                continue
            loss += angle_loss_linear(self.directions[self.direction_idx_curr](z), self.directions[idx](z), self.reg) / (len(self.directions)-1)

        with open(lossfile, 'a') as file:
            file.writelines(str(batch_start)+",mse_loss,"+str(mse_loss)+"\n")
            file.writelines(str(batch_start) + ",overall_loss," + str(loss)+"\n")
            file.writelines(str(batch_start) + ",angle_loss," + str(loss-mse_loss)+"\n")
        return loss


class MultiDirectionZ_linear(nn.Module):
    def __init__(self,dim_z,vocab_size=1000, **kwargs):
        super(MultiDirectionZ_linear, self).__init__()
        self.dim_z = dim_z
        self.vocab_size = vocab_size
        self.direction_num = int(kwargs["direction_num"])
        self.reg = int(kwargs["angle_loss_reg"])
        self.directions = nn.ModuleList([]) # stores all direction parameters

        for _ in range(self.direction_num):
            self.directions.append(nn.Linear(self.dim_z, self.dim_z))
        self.direction_idx_curr = 0  # currently optimized direction index, changes every iteration 

        self.criterion = nn.MSELoss()

    def transform(self,z,y,step_sizes,**kwargs):
        self.direction_idx_curr = (self.direction_idx_curr + 1) % self.direction_num # direction index update
        if y is not None:
            assert(len(y) == z.shape[0])
        interim = step_sizes * self.directions[self.direction_idx_curr](z)
        z_transformed = z + interim
        z_transformed = z.norm() * z_transformed / z_transformed.norm()
        return(z_transformed)
    
    def transform_test(self,z,y,step_sizes,current_way):
        self.direction_idx_curr = current_way # direction index update
        if y is not None:
            assert(len(y) == z.shape[0])
        interim = step_sizes * self.directions[self.direction_idx_curr](z)
        z_transformed = z + interim
        z_transformed = z.norm() * z_transformed / z_transformed.norm()
        return(z_transformed)

    def compute_loss(self, current, target, batch_start, lossfile, z):
        loss = self.criterion(current,target)
        mse_loss = loss.item()
        for idx in range(len(self.directions)):  # calculate the angle losses between the curr direction and all other directions  
            if idx == self.direction_idx_curr:
                continue
            loss += angle_loss_linear(self.directions[self.direction_idx_curr](z), self.directions[idx](z), self.reg) / (len(self.directions)-1)

        with open(lossfile, 'a') as file:
            file.writelines(str(batch_start)+",mse_loss,"+str(mse_loss)+"\n")
            file.writelines(str(batch_start) + ",overall_loss," + str(loss)+"\n")
            file.writelines(str(batch_start) + ",angle_loss," + str(loss-mse_loss)+"\n")
        return loss


class MultiDirectionZ(nn.Module):
    def __init__(self,dim_z,vocab_size=1000, **kwargs):
        super(MultiDirectionZ, self).__init__()
        self.dim_z = dim_z
        self.vocab_size = vocab_size
        self.direction_num = int(kwargs["direction_num"])
        self.reg = int(kwargs["angle_loss_reg"])
        self.directions = nn.ParameterList([]) # stores all direction parameters
        
        for _ in range(self.direction_num):
            self.directions.append(nn.Parameter(torch.randn(1, self.dim_z)))
        self.direction_idx_curr = 0  # currently optimized direction index, changes every iteration 

        self.criterion = nn.MSELoss()

    def transform(self,z,y,step_sizes,**kwargs):
        self.direction_idx_curr = (self.direction_idx_curr + 1) % self.direction_num # direction index update
        if y is not None:
            assert(len(y) == z.shape[0])
        interim = step_sizes * self.directions[self.direction_idx_curr]
        z_transformed = z + interim
        z_transformed = z.norm() * z_transformed / z_transformed.norm()
        return(z_transformed)
    
    
    def transform_test(self,z,y,step_sizes,current_way):
        self.direction_idx_curr = current_way # direction index update
        if y is not None:
            assert(len(y) == z.shape[0])
        interim = step_sizes * self.directions[self.direction_idx_curr]
        z_transformed = z + interim
        z_transformed = z.norm() * z_transformed / z_transformed.norm()
        return(z_transformed)

    def transform_interpolation(self,z,y,step_sizes, values, nonused_i):
        if y is not None:
            assert(len(y) == z.shape[0])

        interpolated_dir = torch.zeros_like(self.directions[0])
        value_i = 0
        for i in range(len(self.directions)):
            if i == nonused_i:
                continue
            interpolated_dir += values[value_i]*self.directions[i]
            value_i += 1
            
        interim = step_sizes * interpolated_dir
        z_transformed = z + interim
        z_transformed = z.norm() * z_transformed / z_transformed.norm()
        return(z_transformed)

    def compute_loss(self, current, target, batch_start, lossfile):
        loss = self.criterion(current,target)
        mse_loss = loss.item()
        for idx in range(len(self.directions)):  # calculate the angle losses between the curr direction and all other directions  
            if idx == self.direction_idx_curr:
                continue
            loss += angle_loss(self.directions[self.direction_idx_curr], self.directions[idx], self.reg) / (len(self.directions)-1)

        with open(lossfile, 'a') as file:
            file.writelines(str(batch_start)+",mse_loss,"+str(mse_loss)+"\n")
            file.writelines(str(batch_start) + ",overall_loss," + str(loss)+"\n")
            file.writelines(str(batch_start) + ",angle_loss," + str(loss-mse_loss)+"\n")
        return loss



class OneDirectionZ(nn.Module):
    def __init__(self,dim_z,vocab_size=1000, **kwargs):
        super(OneDirectionZ, self).__init__()
        self.dim_z = dim_z
        self.vocab_size = vocab_size
        self.w = nn.Parameter(torch.randn(1, self.dim_z))
        self.criterion = nn.MSELoss()

    def transform(self,z,y,step_sizes,**kwargs):
        if y is not None:
            assert(len(y) == z.shape[0])
        interim = step_sizes * self.w
        z_transformed = z + interim
        z_transformed = z.norm() * z_transformed / z_transformed.norm()
        return(z_transformed)

    def compute_loss(self, current, target, batch_start, lossfile):
        loss = self.criterion(current,target)
        with open(lossfile, 'a') as file:
            file.writelines(str(batch_start)+",mse_loss,"+str(loss)+"\n")
            file.writelines(str(batch_start) + ",overall_loss," + str(loss)+"\n")
        return loss


class OneDirectionZAdaptiveY(nn.Module):
    """
    This transformation produces two directions given a class and latent vector.
    The class and latent vectors are updated with these directions.
    """   
    def __init__(self,dim_z,vocab_size=1000, **kwargs):
        super(OneDirectionZAdaptiveY, self).__init__()
        self.dim_z = dim_z
        self.vocab_size = vocab_size
        self.class_reg = int(kwargs["class_reg"])
        self.w = nn.Parameter(torch.randn(1, self.dim_z))
        self.NN_output_y = nn.Linear(self.vocab_size, self.vocab_size)
        self.criterion = nn.MSELoss()

    def transform(self,z,y,step_sizes, step_sizes_y,**kwargs):
        interim_z = step_sizes * self.w
        z_transformed = z + interim_z
        z_transformed = z.norm() * z_transformed / z_transformed.norm()
        interim_y = step_sizes_y * self.NN_output_y(y) / self.class_reg
        y_transformed = y + interim_y
        return z_transformed, y_transformed

    def compute_loss(self, current, target, batch_start, lossfile):
        loss = self.criterion(current,target)
        with open(lossfile, 'a') as file:
            file.writelines(str(batch_start)+",mse_loss,"+str(loss)+"\n")
            file.writelines(str(batch_start) + ",overall_loss," + str(loss)+"\n")
        return loss    
    

class OneDirectionZAdaptiveY_zy(nn.Module):
    """
    This transformation produces two directions given a class and latent vector.
    The class and latent vectors are updated with these directions.
    """   
    def __init__(self,dim_z,vocab_size=1000, **kwargs):
        super(OneDirectionZAdaptiveY_zy, self).__init__()
        self.dim_z = dim_z
        self.vocab_size = vocab_size
        self.class_reg = int(kwargs["class_reg"])
        self.w = nn.Parameter(torch.randn(1, self.dim_z))
        self.NN_output_y = nn.Linear(self.dim_z + self.vocab_size, self.vocab_size)
        self.criterion = nn.MSELoss()

    def transform(self,z,y,step_sizes, step_sizes_y,**kwargs):
        interim_z = step_sizes * self.w
        z_transformed = z + interim_z
        z_transformed = z.norm() * z_transformed / z_transformed.norm()
        interim_y = step_sizes_y * self.NN_output_y(torch.cat((z,y), dim=1)) / self.class_reg
        y_transformed = y + interim_y
        return z_transformed, y_transformed

    def compute_loss(self, current, target, batch_start, lossfile):
        loss = self.criterion(current,target)
        with open(lossfile, 'a') as file:
            file.writelines(str(batch_start)+",mse_loss,"+str(loss)+"\n")
            file.writelines(str(batch_start) + ",overall_loss," + str(loss)+"\n")
        return loss    


class AdaptiveDirectionZ(nn.Module):
    def __init__(self,dim_z,vocab_size=1000, **kwargs):
        super(AdaptiveDirectionZ, self).__init__()
        self.dim_z = dim_z
        self.vocab_size = vocab_size
        self.NN_output = nn.Linear(self.dim_z, self.dim_z)
        self.criterion = nn.MSELoss()

    def transform(self,z,y,step_sizes,**kwargs):
        interim = step_sizes * self.NN_output(z)
        z_transformed = z + interim
        z_transformed = z.norm() * z_transformed / z_transformed.norm()
        return(z_transformed)

    def compute_loss(self, current, target, batch_start, lossfile):
        loss = self.criterion(current,target)
        with open(lossfile, 'a') as file:
            file.writelines(str(batch_start)+",mse_loss,"+str(loss)+"\n")
            file.writelines(str(batch_start) + ",overall_loss," + str(loss)+"\n")
        return loss


class AdaptiveDirectionZAdaptiveDirectionY(nn.Module):
    """
    This transformation produces two directions given a class and latent vector.
    The class and latent vectors are updated with these directions.
    """   
    def __init__(self,dim_z,vocab_size=1000, **kwargs):
        super(AdaptiveDirectionZAdaptiveDirectionY, self).__init__()
        self.dim_z = dim_z
        self.vocab_size = vocab_size
        self.class_reg = int(kwargs["class_reg"])
        if "noise_dim" in kwargs:
            self.noise_dim = int(kwargs["noise_dim"])
            self.y_noise_dim = int(self.noise_dim*(float(vocab_size)/dim_z))
        else:
            self.noise_dim = 0
            self.y_noise_dim = 0
        
        print("Z noise dim:", self.noise_dim)
        print("Y noise dim:", self.y_noise_dim)

        self.class_reg = int(kwargs["class_reg"])
        self.NN_output_z = nn.Linear(self.dim_z+self.noise_dim, self.dim_z)
        self.NN_output_y = nn.Linear(self.vocab_size+self.y_noise_dim, self.vocab_size)
        self.criterion = nn.MSELoss()


    def transform(self,z,y,step_sizes, step_sizes_y, seed=None, **kwargs):
        
        batch_size = z.shape[0]
        if seed:
            torch.random.manual_seed(seed)
        noise_z = torch.randn(batch_size, self.noise_dim).to("cuda")
        interim_z = step_sizes * self.NN_output_z(torch.cat((z, noise_z), dim=1))
        z_transformed = z + interim_z
        z_transformed = z.norm() * z_transformed / z_transformed.norm()

        noise_y = torch.randn(batch_size, self.y_noise_dim).to("cuda")
        interim_y = step_sizes_y * self.NN_output_y(torch.cat((y, noise_y), dim=1)) / self.class_reg
        y_transformed = y + interim_y

        return z_transformed, y_transformed


    def compute_loss(self, current, target, batch_start, lossfile):
        loss = self.criterion(current,target)
        with open(lossfile, 'a') as file:
            file.writelines(str(batch_start)+",mse_loss,"+str(loss)+"\n")
            file.writelines(str(batch_start) + ",overall_loss," + str(loss)+"\n")
        return loss


class AdaptiveDirectionZAdaptiveDirectionY_noise_nonlinear(nn.Module):
    """
    This transformation produces two directions given a class and latent vector.
    The class and latent vectors are updated with these directions.
    """   
    def __init__(self,dim_z,vocab_size=1000, **kwargs):
        super(AdaptiveDirectionZAdaptiveDirectionY_noise_nonlinear, self).__init__()
        self.dim_z = dim_z
        self.vocab_size = vocab_size
        self.class_reg = int(kwargs["class_reg"])
        if "noise_dim" in kwargs:
            self.noise_dim = int(kwargs["noise_dim"])
            self.y_noise_dim = int(self.noise_dim*(float(vocab_size)/dim_z))
        else:
            self.noise_dim = 0
            self.y_noise_dim = 0
        
        print("Z noise dim:", self.noise_dim)
        print("Y noise dim:", self.y_noise_dim)

        self.class_reg = int(kwargs["class_reg"])
        self.NN_output_z = nn.Sequential(nn.Linear(self.dim_z+self.noise_dim, self.dim_z), nn.ReLU(), nn.Linear(self.dim_z, self.dim_z))
        self.NN_output_y = nn.Sequential(nn.Linear(self.vocab_size+self.y_noise_dim, self.vocab_size), nn.ReLU(), nn.Linear(self.vocab_size, self.vocab_size))
        self.criterion = nn.MSELoss()


    def transform(self,z,y,step_sizes, step_sizes_y, seed=None, **kwargs):
        
        batch_size = z.shape[0]
        if seed:
            torch.random.manual_seed(seed)
        noise_z = torch.randn(batch_size, self.noise_dim).to("cuda")
        interim_z = step_sizes * self.NN_output_z(torch.cat((z, noise_z), dim=1))
        z_transformed = z + interim_z
        z_transformed = z.norm() * z_transformed / z_transformed.norm()

        noise_y = torch.randn(batch_size, self.y_noise_dim).to("cuda")
        interim_y = step_sizes_y * self.NN_output_y(torch.cat((y, noise_y), dim=1)) / self.class_reg
        y_transformed = y + interim_y

        return z_transformed, y_transformed


    def compute_loss(self, current, target, batch_start, lossfile):
        loss = self.criterion(current,target)
        with open(lossfile, 'a') as file:
            file.writelines(str(batch_start)+",mse_loss,"+str(loss)+"\n")
            file.writelines(str(batch_start) + ",overall_loss," + str(loss)+"\n")
        return loss
    
    
class AdaptiveDirectionZAdaptiveDirectionY_nonlinear(nn.Module):
    """
    This transformation produces two directions given a class and latent vector.
    The class and latent vectors are updated with these directions.
    """   
    def __init__(self,dim_z,vocab_size=1000, **kwargs):
        super(AdaptiveDirectionZAdaptiveDirectionY_nonlinear, self).__init__()
        self.dim_z = dim_z
        self.vocab_size = vocab_size
        self.class_reg = int(kwargs["class_reg"])
        if "noise_dim" in kwargs:
            self.noise_dim = int(kwargs["noise_dim"])
            self.y_noise_dim = int(self.noise_dim*(float(vocab_size)/dim_z))
        else:
            self.noise_dim = 0
            self.y_noise_dim = 0
        
        print("Z noise dim:", self.noise_dim)
        print("Y noise dim:", self.y_noise_dim)

        self.class_reg = int(kwargs["class_reg"])
        self.NN_output_z = nn.Sequential(nn.Linear(self.dim_z, self.dim_z), nn.ReLU(), nn.Linear(self.dim_z, self.dim_z))
        self.NN_output_y = nn.Sequential(nn.Linear(self.vocab_size, self.vocab_size), nn.ReLU(), nn.Linear(self.vocab_size, self.vocab_size))
        self.criterion = nn.MSELoss()


    def transform(self,z,y,step_sizes, step_sizes_y, seed=None, **kwargs):
        
        batch_size = z.shape[0]
        
        
        interim_z = step_sizes * self.NN_output_z(z)
        z_transformed = z + interim_z
        z_transformed = z.norm() * z_transformed / z_transformed.norm()

        
        interim_y = step_sizes_y * self.NN_output_y(y) / self.class_reg
        y_transformed = y + interim_y

        return z_transformed, y_transformed


    def compute_loss(self, current, target, batch_start, lossfile):
        loss = self.criterion(current,target)
        with open(lossfile, 'a') as file:
            file.writelines(str(batch_start)+",mse_loss,"+str(loss)+"\n")
            file.writelines(str(batch_start) + ",overall_loss," + str(loss)+"\n")
        return loss    

class AdaptiveDirectionZAdaptiveDirectionY_zy(nn.Module):
    """
    This transformation produces two directions given a class and latent vector.
    The class and latent vectors are updated with these directions.
    """   
    def __init__(self,dim_z,vocab_size=1000, **kwargs):
        super(AdaptiveDirectionZAdaptiveDirectionY_zy, self).__init__()
        self.dim_z = dim_z
        self.vocab_size = vocab_size
        self.class_reg = int(kwargs["class_reg"])
        if "noise_dim" in kwargs:
            self.noise_dim = int(kwargs["noise_dim"])
            self.y_noise_dim = int(self.noise_dim*(float(vocab_size)/dim_z))
        else:
            self.noise_dim = 0
            self.y_noise_dim = 0
        
        print("Z noise dim:", self.noise_dim)
        print("Y noise dim:", self.y_noise_dim)

        self.NN_output_z = nn.Linear(self.dim_z+self.noise_dim, self.dim_z)
        self.NN_output_y = nn.Linear(self.vocab_size+self.y_noise_dim+self.dim_z, self.vocab_size)
        self.criterion = nn.MSELoss()

    def transform(self,z,y,step_sizes, step_sizes_y, seed=None, **kwargs):
        
        batch_size = z.shape[0]
        if seed:
            torch.random.manual_seed(seed)
        noise_z = torch.randn(batch_size, self.noise_dim).to("cuda")
        interim_z = step_sizes * self.NN_output_z(torch.cat((z, noise_z), dim=1))
        z_transformed = z + interim_z
        z_transformed = z.norm() * z_transformed / z_transformed.norm()

        noise_y = torch.randn(batch_size, self.y_noise_dim).to("cuda")
        interim_y = step_sizes_y * self.NN_output_y(torch.cat((z,y, noise_y), dim=1)) / self.class_reg
        y_transformed = y + interim_y
        return z_transformed, y_transformed

    def compute_loss(self, current, target, batch_start, lossfile):
        loss = self.criterion(current,target)
        with open(lossfile, 'a') as file:
            file.writelines(str(batch_start)+",mse_loss,"+str(loss)+"\n")
            file.writelines(str(batch_start) + ",overall_loss," + str(loss)+"\n")
        return loss


class AdaptiveDirectionZ_zy(nn.Module):
    """
    This transformation produces a direction given a class and latent vector. Produced
    direction only affects the latent vector
    """
    def __init__(self,dim_z,vocab_size=1000, **kwargs):
        super(AdaptiveDirectionZ_zy, self).__init__()
        self.dim_z = dim_z
        self.vocab_size = vocab_size
        self.NN_output = nn.Linear(self.dim_z+self.vocab_size, self.dim_z)
        self.criterion = nn.MSELoss()


    def transform(self,z,y,step_sizes,**kwargs):
        interim = step_sizes * self.NN_output(torch.cat((z,y), dim=1))
        z_transformed = z + interim
        z_transformed = z.norm() * z_transformed / z_transformed.norm()
        return(z_transformed)

    def compute_loss(self, current, target, batch_start, lossfile):
        loss = self.criterion(current,target)
        with open(lossfile, 'a') as file:
            file.writelines(str(batch_start)+",mse_loss,"+str(loss)+"\n")
            file.writelines(str(batch_start) + ",overall_loss," + str(loss)+"\n")
        return loss


class AdaptiveDirectionZ_zy_AdaptiveDirectionY(nn.Module):
    """
    This transformation produces a direction given a class and latent vector. Produced
    direction only affects the latent vector
    """
    def __init__(self,dim_z,vocab_size=1000, **kwargs):
        super(AdaptiveDirectionZ_zy_AdaptiveDirectionY, self).__init__()
        self.dim_z = dim_z
        self.vocab_size = vocab_size
        self.class_reg = int(kwargs["class_reg"])
        self.NN_output = nn.Linear(self.dim_z+self.vocab_size, self.dim_z)
        self.NN_output_y = nn.Linear(self.vocab_size, self.vocab_size)
        self.criterion = nn.MSELoss()

    def transform(self,z,y,step_sizes, step_sizes_y,**kwargs):
        interim = step_sizes * self.NN_output(torch.cat((z,y), dim=1))
        z_transformed = z + interim
        z_transformed = z.norm() * z_transformed / z_transformed.norm()
        interim_y = step_sizes_y * self.NN_output_y(y) / self.class_reg
        y_transformed = y + interim_y
        return z_transformed, y_transformed

    def compute_loss(self, current, target, batch_start, lossfile):
        loss = self.criterion(current,target)
        with open(lossfile, 'a') as file:
            file.writelines(str(batch_start)+",mse_loss,"+str(loss)+"\n")
            file.writelines(str(batch_start) + ",overall_loss," + str(loss)+"\n")
        return loss


class AdaptiveDirectionZ_zy_AdaptiveDirectionY_zy(nn.Module):
    """
    This transformation produces a direction given a class and latent vector. Produced
    direction only affects the latent vector
    """
    def __init__(self,dim_z,vocab_size=1000, **kwargs):
        super(AdaptiveDirectionZ_zy_AdaptiveDirectionY_zy, self).__init__()
        self.dim_z = dim_z
        self.vocab_size = vocab_size
        self.class_reg = int(kwargs["class_reg"])
        self.NN_output = nn.Linear(self.dim_z+self.vocab_size, self.dim_z)
        self.NN_output_y = nn.Linear(self.dim_z+self.vocab_size, self.vocab_size)
        self.criterion = nn.MSELoss()

    def transform(self,z,y,step_sizes, step_sizes_y,**kwargs):
        interim = step_sizes * self.NN_output(torch.cat((z,y), dim=1))
        z_transformed = z + interim
        z_transformed = z.norm() * z_transformed / z_transformed.norm()
        interim_y = step_sizes_y * self.NN_output_y(torch.cat((z,y), dim=1)) / self.class_reg
        y_transformed = y + interim_y
        return z_transformed, y_transformed

    def compute_loss(self, current, target, batch_start, lossfile):
        loss = self.criterion(current,target)
        with open(lossfile, 'a') as file:
            file.writelines(str(batch_start)+",mse_loss,"+str(loss)+"\n")
            file.writelines(str(batch_start) + ",overall_loss," + str(loss)+"\n")
        return loss


class ClassDependent(nn.Module):
    def __init__(self,dim_z,vocab_size=1000, **kwargs):
        super(ClassDependent, self).__init__()
        print("\napproach: ", "class_dependent\n")
        self.dim_z = dim_z
        self.vocab_size = vocab_size
        self.NN_output = nn.Linear(self.vocab_size, self.dim_z)
        self.criterion = nn.MSELoss()


    def transform(self,z,y,step_sizes,**kwargs):
        assert (y is not None)
        interim = step_sizes * self.NN_output(y)
        z_transformed = z + interim
        z_transformed = z.norm() * z_transformed / z_transformed.norm()
        return(z_transformed)

    def compute_loss(self, current, target, batch_start, lossfile):
        loss = self.criterion(current,target)
        with open(lossfile, 'a') as file:
            file.writelines(str(batch_start)+",mse_loss,"+str(loss)+"\n")
            file.writelines(str(batch_start) + ",overall_loss," + str(loss)+"\n")
        return loss



class AdaptiveDirectionZ_nonlinear(nn.Module):
    def __init__(self,dim_z,vocab_size=1000, **kwargs):
        super(AdaptiveDirectionZ_nonlinear, self).__init__()
        self.dim_z = dim_z
        self.vocab_size = vocab_size
        self.NN_output = nn.Linear(self.dim_z, self.dim_z)
        self.NN_output2 = nn.Linear(self.dim_z, self.dim_z)
        self.relu = nn.ReLU()
        self.criterion = nn.MSELoss()

    def transform(self,z,y,step_sizes,**kwargs):
        interim = step_sizes *  self.NN_output2(self.relu(self.NN_output(z)))
        z_transformed = z + interim
        z_transformed = z.norm() * z_transformed / z_transformed.norm()
        return(z_transformed)

    def compute_loss(self, current, target, batch_start, lossfile):
        loss = self.criterion(current,target)
        with open(lossfile, 'a') as file:
            file.writelines(str(batch_start)+",mse_loss,"+str(loss)+"\n")
            file.writelines(str(batch_start) + ",overall_loss," + str(loss)+"\n")
        return loss    
