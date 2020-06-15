
import numpy as np

#------------------------------------------------------------------------#
def ncon_solver(tensors,connects,max_kept=None):

  # start by trying both greed algorithms
  log_adj, costs, groups, orders = ncon_to_adj(tensors,connects)
  order0, cost0 = greedy_size_solve(log_adj.copy())
  order1, cost1 = greedy_cost_solve(log_adj.copy())
  if (cost0 < cost1):
    order_greedy = order0
    cost_greedy = cost0
  else:
    order_greedy = order1
    cost_greedy = cost1
  
  if (max_kept == 1):
    # return results from greedy
    order_was_found = False
  else:
    # try full algorithm (using cost_bound from greedy)
    cost_bound = cost_greedy + 1e-4 #small fudge factor
    N = log_adj.shape[0]
    total_truncated = 0
    num_samples = 1
    order_was_found = True
    
    for t in range(N-1):
      log_adj,costs,groups,orders,num_truncated = full_solve_order(log_adj,costs,groups,orders,cost_bound=cost_bound,max_kept=max_kept)
      if (log_adj.size == 0):
        order_was_found = False
        break    
      total_truncated = total_truncated + num_truncated
      num_samples = np.append(num_samples,costs.size)
  
  if order_was_found:
    cont_order = ord_to_ncon(connects,orders)
    is_optimal = (total_truncated == 0)
    return cont_order, costs.item(), is_optimal
  else:
    cont_order = ord_to_ncon(connects,order_greedy)
    is_optimal = False
    return cont_order, cost_greedy, is_optimal
  
#------------------------------------------------------------------------#
def greedy_size_solve(log_adj):
  N = log_adj.shape[0]
  log_adj = log_adj.reshape(N,N)
  orders = np.zeros([2,0],dtype=int)
  costs = 0;
  
  for t in range(N-1):
    # compute tensor dims
    N = log_adj.shape[0]
    dims = np.sum(log_adj,axis=0).reshape(N)
    comb_dims = np.add.outer(dims,dims)
    
    # compute contraction costs and new dims
    single_cost = comb_dims - log_adj
    new_dims = comb_dims - 2*log_adj
    new_dims = new_dims + max(new_dims.flatten())*np.eye(N)
    
    # compute maximum dim of tensor in contraction 
    temp_mat = np.kron(dims,np.ones([N,1]))
    max_dim = np.maximum(temp_mat,temp_mat.T)
    
    dim_change = (1000*(new_dims - max_dim)).astype(int)
     # dim_change = dim_change + max(dim_change.flatten())*np.eye(N)
    
    # compute coords of minimal dim increase 
    xcoord, ycoord = np.where(dim_change == min(dim_change.flatten()))
    upper_tri = (xcoord < ycoord)
    xcoord = xcoord[upper_tri]
    ycoord = ycoord[upper_tri]
    
    # find contraction with minimal cost
    all_costs = [single_cost[xcoord[i],ycoord[i]] for i in range(len(xcoord))]
    cheapest_pos = np.argmin(all_costs)
    i = ycoord[cheapest_pos]
    j = xcoord[cheapest_pos]
    
    # build new log adjacency
    log_adj[j,j] = log_adj[j,j] - 2*log_adj[j,i] 
    log_adj[j,:] = log_adj[j,:] + log_adj[i,:]
    log_adj[:,j] = log_adj[:,j] + log_adj[:,i]
    log_adj = np.delete(log_adj, i, axis=0)
    log_adj = np.delete(log_adj, i, axis=1)
    
    # build new orders
    orders = np.hstack((orders,np.asarray([j,i]).reshape(2,1)))
    
    # tally the cost
    costs = np.log10(10**costs + 10**single_cost[i,j])
    
  return orders, costs

#------------------------------------------------------------------------#
def greedy_cost_solve(log_adj):
  N = log_adj.shape[0]
  log_adj = log_adj.reshape(N,N)
  orders = np.zeros([2,0],dtype=int)
  costs = 0;
  
  for t in range(N-1):
    # compute tensor dims and costs
    N = log_adj.shape[0]
    dims = np.sum(log_adj,axis=0).reshape(N)
    comb_dims = np.add.outer(dims,dims) 
    single_cost = comb_dims - log_adj
    
    # penalize trivial contractions and self-contractions
    triv_conts = (log_adj < 0.001) # small fudge factor 
    trimmed_costs = single_cost + max(single_cost.flatten())*triv_conts
    trimmed_costs = trimmed_costs + max(trimmed_costs.flatten())*np.eye(N)
    
    # find best contraction
    tensors_to_contract = np.divmod(np.argmin(trimmed_costs),N)
    i = max(tensors_to_contract)
    j = min(tensors_to_contract)
    
    # build new log adjacency
    log_adj[j,j] = log_adj[j,j] - 2*log_adj[j,i] 
    log_adj[j,:] = log_adj[j,:] + log_adj[i,:]
    log_adj[:,j] = log_adj[:,j] + log_adj[:,i]
    log_adj = np.delete(log_adj, i, axis=0)
    log_adj = np.delete(log_adj, i, axis=1)
    
    # build new orders
    orders = np.hstack((orders,np.asarray(tensors_to_contract).reshape(2,1)))
    
    # tally the cost
    costs = np.log10(10**costs + 10**single_cost[i,j])
  
  return orders, costs

#------------------------------------------------------------------------#
def full_solve_order(log_adj,costs,groups,orders,cost_bound=None,max_kept=None,allow_outer=False):
  
  # set threshold required to trigger compression routine
  if max_kept is None:
    mid_kept = 10000
  else:
    mid_kept = max_kept
  
  # initialize outputs
  N = log_adj.shape[0]
  final_adj = np.zeros([N-1,N-1,0])
  final_costs = np.zeros([1,0])
  final_groups = np.zeros([N-1,0],dtype=np.uint64)
  final_orders = np.zeros([2,orders.shape[1]+1,0],dtype=int)
  final_stable = np.zeros([1,0],dtype=bool)
  total_truncated = 0
  
  only_outer_exist = not allow_outer
  none_inbounds = True
  
  # try to contract j-th tensor with i-th tensor (j<i)
  for i in range(1,N):
    for j in range(i):
      
      # only attempt non-trivial contractions
      new_pos = np.flatnonzero(log_adj[j,i,:] > 0) 
      num_kept = len(new_pos)
      
      if num_kept > 0:
        only_outer_exist = False
        
        # dims of tensors and cost od contraction
        dims = np.sum(log_adj[:,:,new_pos],axis=0).reshape(N,num_kept)
        comb_dims = dims[j,:] + dims[i,:]
        single_cost = np.reshape(comb_dims - log_adj[j,i,new_pos],[1,num_kept])
        new_costs = np.log10(10**costs[0,new_pos] + 10**single_cost)
        
        if cost_bound is not None:
          # only keep contractions under the cost bound
          pos_under_bound = new_costs.flatten() < cost_bound
          new_pos = new_pos[pos_under_bound]
          num_kept = len(new_pos)
          
          new_costs = new_costs[0,pos_under_bound].reshape(1,num_kept)
        
      if num_kept > 0:
        none_inbounds = False
        
        # order the costs
        cost_order = np.argsort(new_costs).flatten();
        sorted_pos = new_pos[cost_order]
        
        # identify identical networks
        new_groups = groups[:,sorted_pos]
        new_groups[j,:] = new_groups[j,:] + new_groups[i,:]
        new_groups = np.delete(new_groups, i, axis=0)
        new_groups,temp_pos = np.unique(new_groups,return_index=True,axis=1)
        
        new_costs = new_costs[:,cost_order[temp_pos]]
        new_pos = sorted_pos[temp_pos]
        num_kept = len(new_pos)
        
        # new log adjacency
        new_adj = log_adj[:,:,new_pos]
        new_adj[j,j,:] = new_adj[j,j,:] - 2*new_adj[j,i,:] 
        new_adj[j,:,:] = new_adj[j,:,:] + new_adj[i,:,:]
        new_adj[:,j,:] = new_adj[:,j,:] + new_adj[:,i,:]
        new_adj = np.delete(new_adj, i, axis=0)
        new_adj = np.delete(new_adj, i, axis=1)
        
        # new orders
        prev_orders = orders[:,:,new_pos]
        next_orders = np.vstack([j*np.ones(len(new_pos),dtype=int),i*np.ones(len(new_pos),dtype=int)]).reshape(2,1,len(new_pos))
        new_orders = np.concatenate((prev_orders,next_orders),axis=1)
        
        # new_stable
        dims = np.sum(log_adj[:,:,new_pos],axis=0).reshape(N,num_kept)
        comb_dims = dims[j,:] + dims[i,:]
        final_dims = np.reshape(comb_dims - 2*log_adj[j,i,new_pos],[1,num_kept])
        stable_pos = final_dims < (np.maximum(dims[j,:],dims[i,:])+0.001) # includes fudge factor to avoid rounding errors 
        
        final_adj = np.concatenate((final_adj,new_adj),axis=2)
        final_costs = np.concatenate((final_costs,new_costs),axis=1)
        final_groups = np.concatenate((final_groups,new_groups),axis=1)
        final_orders = np.concatenate((final_orders,new_orders),axis=2)
        final_stable = np.concatenate((final_stable,stable_pos),axis=1)
        
        # if number of intermediates too large then trigger compression routine
        if final_costs.size > mid_kept:
          temp_pos, num_truncated = reduce_nets(final_costs,final_groups,final_stable,max_kept=max_kept)
          final_adj = final_adj[:,:,temp_pos]
          final_costs = final_costs[:,temp_pos]
          final_groups = final_groups[:,temp_pos]
          final_orders = final_orders[:,:,temp_pos]
          final_stable = final_stable[:,temp_pos]
          total_truncated = total_truncated + num_truncated
  
  if not only_outer_exist:
    if none_inbounds:
      # no orders found under the cost bound; return trivial
      return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), 0
  
  if only_outer_exist: # network contains only outer products
    # re-solve with outer products enabled
    return full_solve_order(log_adj,costs,groups,orders,cost_bound=cost_bound,max_kept=max_kept,allow_outer=True)
  else: 
    # compress outputs
    temp_pos = reduce_nets(final_costs,final_groups,final_stable)[0]
    final_adj = final_adj[:,:,temp_pos]
    final_costs = final_costs[:,temp_pos]
    final_groups = final_groups[:,temp_pos]
    final_orders = final_orders[:,:,temp_pos]
    final_stable = final_stable[:,temp_pos]
    return final_adj,final_costs,final_groups,final_orders,total_truncated
        
#------------------------------------------------------------------------#
def reduce_nets(costs,groups,stable,max_kept=None):
  
  # sort according to the costs
  new_pos = np.argsort(costs).flatten();

  # identify identical networks
  temp_pos = np.unique(groups[:,new_pos],return_index=True,axis=1)[1]
  orig_kept = len(temp_pos)
  
  new_pos = new_pos[temp_pos]
  num_truncated = 0;
  
  if max_kept is not None:
    if (orig_kept > max_kept):
      # re-sort according to the costs (np.unique breaks cost ordering
      cost_order = np.argsort(costs[:,new_pos]).flatten()
      new_pos = new_pos[cost_order]
      
      # reserve some percertage for size-stable contractions
      preserve_ratio = 0.2;
      num_stable = int(np.ceil(max_kept*preserve_ratio))
      num_cheapest = int(np.ceil(max_kept*(1-preserve_ratio)))
      
      stable_pos = np.flatnonzero(stable[0,new_pos[num_cheapest:]])
      temp_pos = np.concatenate((np.arange(num_cheapest),stable_pos[:min(len(stable_pos),num_stable)] + num_cheapest),axis=0)
      
      new_pos = new_pos[temp_pos]
      num_truncated = orig_kept-len(new_pos)
      
  return new_pos, num_truncated

#------------------------------------------------------------------------#
def ncon_to_adj(tensors,connects):
  N = len(connects)
  ranks = [len(connects[i]) for i in range(N)]
  flat_connects = np.hstack([connects[i] for i in range(N)])
  tensor_counter = np.hstack([i*np.ones(ranks[i],dtype=int) for i in range(N)])
  index_counter = np.hstack([np.arange(ranks[i]) for i in range(N)])
  
  log_adj = np.zeros([N,N]);
  unique_connects = np.unique(flat_connects)
  for i in range(len(unique_connects)):
      t_loc = tensor_counter[flat_connects == unique_connects[i]]
      i_loc = index_counter[flat_connects == unique_connects[i]]
      if len(i_loc) == 1: # external index
        log_adj[t_loc[0],t_loc[0]] = log_adj[t_loc[0],t_loc[0]] + np.log10(tensors[t_loc[0]].shape[i_loc[0]])
      elif len(i_loc) == 2: # internal index
        log_adj[t_loc[0],t_loc[1]] = log_adj[t_loc[0],t_loc[1]] + np.log10(tensors[t_loc[0]].shape[i_loc[0]])
        log_adj[t_loc[1],t_loc[0]] = log_adj[t_loc[1],t_loc[0]] + np.log10(tensors[t_loc[0]].shape[i_loc[0]])
   
  log_adj = log_adj.reshape(N,N,1)
  costs = np.zeros([1,1])
  groups =  np.array(2**np.arange(N),dtype=np.uint64).reshape(N,1)
  orders = np.zeros([2,0,1],dtype=int)
  
  return log_adj,costs,groups,orders

#------------------------------------------------------------------------#
def ord_to_ncon(connects,orders):
  N = orders.size//2
  orders = orders.reshape(2,N)
  new_connects = [connects[i] for i in range(len(connects))]
  cont_order = np.zeros([0],dtype=int)
  for i in range(N):  
    cont_many, A_cont, B_cont = np.intersect1d(new_connects[orders[0,i]],new_connects[orders[1,i]], assume_unique=True, return_indices=True)
    temp_connects = np.append(np.delete(new_connects[orders[0,i]], A_cont), np.delete(new_connects[orders[1,i]], B_cont))
  
    common_inds = np.intersect1d(new_connects[orders[0,i]],new_connects[orders[1,i]])
    cont_order = np.concatenate((cont_order,common_inds),axis=0)
    
    del new_connects[orders[1,i]]
    new_connects[orders[0,i]] = temp_connects

  return cont_order


