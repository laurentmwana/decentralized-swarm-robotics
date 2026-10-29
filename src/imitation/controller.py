"""Adaptateur réellement local du GAT pour les épisodes à accélérations."""
import numpy as np
import torch


class LocalGATController:
    def __init__(self, model, mission):
        self.model=model.eval()
        self.mission=mission

    def __call__(self, observation):
        # Étoile locale : soi et voisins directs, aucun voisin de voisin.
        positions=np.vstack((observation.position,observation.position+observation.relative_positions))
        velocities=np.vstack((observation.velocity,observation.velocity+observation.relative_velocities))
        n=len(positions)
        features=np.column_stack((positions,velocities,np.tile(self.mission.center,(n,1)),
                                  np.full(n,self.mission.radius),np.full(n,self.mission.robot_count)))
        adjacency=np.zeros((n,n),dtype=bool)
        adjacency[0,1:]=True; adjacency[1:,0]=True
        parameter=next(self.model.parameters())
        with torch.inference_mode():
            command=self.model(torch.as_tensor(features,dtype=parameter.dtype,device=parameter.device),
                               torch.as_tensor(adjacency,device=parameter.device))[0]
        return command.cpu().numpy().astype(float,copy=True)
