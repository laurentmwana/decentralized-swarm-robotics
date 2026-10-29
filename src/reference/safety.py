"""Projection 2D bornée de contraintes locales réciproques d'accélération.

Le filtre est testé dans le simulateur discret ; aucune garantie continue ou
universelle n'est déduite lorsque les contraintes sont infaisables ou écrêtées.
"""
import numpy as np


def project_acceleration(nominal, normals, bounds, maximum):
    """Minimiser ||a-nominal||² avec n.a>=b et ||a||<=maximum ; None si infaisable."""
    nominal = np.asarray(nominal, dtype=float)
    normals = np.asarray(normals, dtype=float).reshape(-1, 2)
    bounds = np.asarray(bounds, dtype=float)
    if nominal.shape != (2,) or bounds.shape != (len(normals),) or not np.isfinite(maximum) or maximum <= 0:
        raise ValueError('Projection d’accélération : dimensions ou limite invalides')
    if not np.isfinite(nominal).all() or not np.isfinite(normals).all() or not np.isfinite(bounds).all():
        raise ValueError('Projection finie requise')
    lengths = np.linalg.norm(normals, axis=1)
    if (lengths <= 1e-12).any():
        raise ValueError('Normales non nulles requises')
    normals = normals/lengths[:, None]; bounds = bounds/lengths
    # Une commande déjà admissible est l'unique minimum, de distance nulle.
    ball_projection=nominal*min(1.,maximum/max(np.linalg.norm(nominal),1e-12))
    if (normals@ball_projection>=bounds).all():
        return ball_projection.copy()
    if (bounds > maximum+1e-10).any(): return None
    active = bounds > -maximum
    normals, bounds = normals[active], bounds[active]
    candidates = [nominal*min(1., maximum/max(np.linalg.norm(nominal), 1e-12))]
    for normal, bound in zip(normals, bounds):
        tangent = np.array([-normal[1], normal[0]])
        extent = np.sqrt(max(0., maximum**2-bound**2))
        candidates.append(bound*normal+np.clip(np.dot(tangent, nominal), -extent, extent)*tangent)
    for i in range(len(normals)):
        for j in range(i):
            matrix = normals[[i, j]]
            if abs(np.linalg.det(matrix)) > 1e-12:
                candidates.append(np.linalg.solve(matrix, bounds[[i, j]]))
    feasible = [a for a in candidates if np.linalg.norm(a) <= maximum+1e-9 and (normals@a >= bounds-1e-9).all()]
    return min(feasible, key=lambda a: np.sum((a-nominal)**2)) if feasible else None


def filter_command(observation, nominal, gains, maximum, *, obstacle_vectors=(),wall_half_extent=None):
    """Une moitié symétrique de chaque contrainte pour chacun des deux robots."""
    normals, bounds = [], []
    nominal = np.array(nominal, dtype=float, copy=True)
    base = nominal.copy()
    fallback = -gains.damping*observation.velocity
    fallback *= min(1., maximum/max(np.linalg.norm(fallback), 1e-12))
    obstacles=np.asarray(obstacle_vectors,dtype=float).reshape(-1,3)
    if not np.isfinite(obstacles).all() or (obstacles[:,2]<=0).any():
        raise ValueError('Obstacles locaux finis et rayons gonflés positifs requis')
    if wall_half_extent is not None:
        if not np.isfinite(wall_half_extent) or wall_half_extent<=0:
            raise ValueError('Limite intérieure des murs finie positive requise')
        # Linear wall distance h, with h'' + 2 h' + h >= 0.
        # Slower wall response starts braking early under bounded acceleration.
        for normal in (np.array([1.,0.]),np.array([-1.,0.]),np.array([0.,1.]),np.array([0.,-1.])):
            gap=wall_half_extent+normal@observation.position
            normals.append(normal);bounds.append(-2*(normal@observation.velocity)-gap)
    for relative, relative_velocity in zip(observation.relative_positions, observation.relative_velocities):
        r, w = -relative, -relative_velocity
        distance = np.linalg.norm(r)
        if distance <= 1e-12: return fallback, False
        h = distance**2-gains.safety_distance**2
        bound = -.5*np.dot(w, w)-gains.barrier_alpha*np.dot(r, w)-.25*gains.barrier_alpha**2*h
        normals.append(r); bounds.append(bound)
        # Même chiralité chez les deux robots pour départager un face-à-face.
        need_detour = not gains.detour_only_on_constraint or np.dot(base, r) < bound-1e-12
        if need_detour and distance < gains.detour_distance and np.dot(base, r) < 0:
            nominal += gains.detour_gain*(1-distance/gains.detour_distance)*np.array([-r[1], r[0]])/distance
    for obstacle in obstacles:
        r=-obstacle[:2];distance=np.linalg.norm(r);radius=obstacle[2]
        if distance<=1e-12:return fallback,False
        w=observation.velocity
        h=distance**2-radius**2
        # Obstacle immobile : la contrainte entière revient au robot mobile.
        bound=-np.dot(w,w)-2*gains.barrier_alpha*np.dot(r,w)-.5*gains.barrier_alpha**2*h
        normals.append(r);bounds.append(bound)
        gap=distance-radius
        need_detour=not gains.detour_only_on_constraint or np.dot(base,r)<bound-1e-12
        if need_detour and gap<gains.detour_distance and np.dot(base,r)<0:
            nominal+=gains.detour_gain*(1-max(gap,0.)/gains.detour_distance)*np.array([-r[1],r[0]])/distance
    projected = project_acceleration(nominal, normals, bounds, maximum)
    return (fallback, False) if projected is None else (projected, True)
