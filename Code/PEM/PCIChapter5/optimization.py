import time
import random
import math

people = [('Seymour','BOS'),
          ('Franny','DAL'),
          ('Zooey','CAK'),
          ('Walt','MIA'),
          ('Buddy','ORD'),
          ('Les','OMA')]
# Laguardia
destination='LGA'

flights={}
# 
# for line in file('schedule.txt'):
with open('schedule.txt','r') as file:
    for line in file:
          origin,dest,depart,arrive,price=line.strip().split(',')
          flights.setdefault((origin,dest),[])

          # Add details to the list of possible flights
          flights[(origin,dest)].append((depart,arrive,int(price)))

def getminutes(t):
  x=time.strptime(t,'%H:%M')
  return x[3]*60+x[4]

def printschedule(r):
  for d in range(len(r)//2):#using integer division
    name = people[d][0]
    origin = people[d][1]
    out = flights[(origin,destination)][int(r[d])]
    ret = flights[(destination,origin)][int(r[d+1])]
    #print '%10s%10s %5s-%5s $%3s %5s-%5s $%3s' % (name,origin,
    #                                              out[0],out[1],out[2],
    #                                              ret[0],ret[1],ret[2])
    print(f"{name:10}{origin:10} {out[0]:5}-{out[1]:5} ${out[2]:3} {ret[0]:5}-{ret[1]:5} ${ret[2]:3}")

def schedulecost(sol):
  totalprice=0
  latestarrival=0
  earliestdep=24*60

  for d in range(len(sol)/2):
    # Get the inbound and outbound flights
    origin=people[d][1]
    outbound=flights[(origin,destination)][int(sol[d])]
    returnf=flights[(destination,origin)][int(sol[d+1])]
    
    # Total price is the price of all outbound and return flights
    totalprice+=outbound[2]
    totalprice+=returnf[2]
    
    # Track the latest arrival and earliest departure
    if latestarrival<getminutes(outbound[1]): latestarrival=getminutes(outbound[1])
    if earliestdep>getminutes(returnf[0]): earliestdep=getminutes(returnf[0])
  
  # Every person must wait at the airport until the latest person arrives.
  # They also must arrive at the same time and wait for their flights.
  totalwait=0  
  for d in range(len(sol)/2):
    origin=people[d][1]
    outbound=flights[(origin,destination)][int(sol[d])]
    returnf=flights[(destination,origin)][int(sol[d+1])]
    totalwait+=latestarrival-getminutes(outbound[1])
    totalwait+=getminutes(returnf[0])-earliestdep  

  # Does this solution require an extra day of car rental? That'll be $50!
  if latestarrival>earliestdep: totalprice+=50
  
  return totalprice+totalwait

def randomoptimize(domain,costf):
  best = 999999999
  #best = float('inf')  
  bestr = None
  for i in range(0,1000):
    # Create a random solution
    r = [int(random.randint(domain[i][0],domain[i][1])) for i in range(len(domain))]
    
    # Get the cost
    cost = costf(r)
    
    # Compare it to the best one so far
    if cost < best:
      best = cost
      bestr = r 
  #return r
  return bestr

def hillclimb(domain,costf):
  # Create a random solution
  sol=[random.randint(domain[i][0],domain[i][1])
      for i in range(len(domain))]
  # Main loop
  while 1:
    # Create list of neighboring solutions
    neighbors=[]
    
    for j in range(len(domain)):
      # One away in each direction
      if sol[j]>domain[j][0]:
        neighbors.append(sol[0:j]+[sol[j]+1]+sol[j+1:])
      if sol[j]<domain[j][1]:
        neighbors.append(sol[0:j]+[sol[j]-1]+sol[j+1:])

    # See what the best solution amongst the neighbors is
    current=costf(sol)
    best=current
    for j in range(len(neighbors)):
      cost=costf(neighbors[j])
      if cost<best:
        best=cost
        sol=neighbors[j]

    # If there's no improvement, then we've reached the top
    if best==current:
      break
  return sol

def annealingoptimize(domain,costf,T=10000.0,cool=0.95,step=1):
  # Initialize the values randomly
  vec=[float(random.randint(domain[i][0],domain[i][1])) 
       for i in range(len(domain))]
  
  while T>0.1:
    # Choose one of the indices
    i=random.randint(0,len(domain)-1)

    # Choose a direction to change it
    dir=random.randint(-step,step)

    # Create a new list with one of the values changed
    vecb=vec[:]
    vecb[i]+=dir
    if vecb[i]<domain[i][0]: vecb[i]=domain[i][0]
    elif vecb[i]>domain[i][1]: vecb[i]=domain[i][1]

    # Calculate the current cost and the new cost
    ea=costf(vec)
    eb=costf(vecb)
    p=pow(math.e,(-eb-ea)/T)

    # Is it better, or does it make the probability
    # cutoff?
    if (eb<ea or random.random()<p):
      vec=vecb      

    # Decrease the temperature
    T=T*cool
  return vec

def geneticoptimize(domain,costf,popsize=50,step=1,
                    mutprob=0.2,elite=0.2,maxiter=100):
  # Mutation Operation 变异操作
  def mutate(vec):
    #i = random.randint(0,len(domain)-1)
    #if random.random()<0.5 and vec[i]>domain[i][0]:
      #return vec[0:i]+[vec[i]-step]+vec[i+1:] 
    #elif vec[i]<domain[i][1]:
      #return vec[0:i]+[vec[i]+step]+vec[i+1:]
    i = random.randint(0, len(domain)-1)
    if random.random() < 0.5:
        if vec[i] > domain[i][0]:
            return vec[0:i] + [vec[i]-step] + vec[i+1:]
        elif vec[i] < domain[i][1]:
            return vec[0:i] + [vec[i]+step] + vec[i+1:]
    else:
        if vec[i] < domain[i][1]:
            return vec[0:i] + [vec[i]+step] + vec[i+1:]
        elif vec[i] > domain[i][0]:
            return vec[0:i] + [vec[i]-step] + vec[i+1:]
    return vec  # Return original if no mutation possible
  
  # Crossover Operation 交叉操作
  def crossover(r1,r2):
    i=random.randint(1,len(domain)-2)
    return r1[0:i]+r2[i:]

  # Build the initial population 构造初始种群
  pop=[]
  for i in range(popsize):
    vec=[random.randint(domain[i][0],domain[i][1]) 
         for i in range(len(domain))]
    pop.append(vec)
  
  # How many winners from each generation? 每一代中有多少胜出者？
  topelite=int(elite*popsize)
  
  # Main loop 主循环
  for i in range(maxiter):
    #print(f"Iteration {i}: Population size = {len(pop)}")
    scores=[(costf(v),v) for v in pop]
    scores.sort()
    ranked=[v for (s,v) in scores]
    
    # Start with the pure winners 从纯粹的胜出者开始
    #print(f"Scores: {scores}")  
    pop=ranked[0:topelite]
    
    # Add mutated and bred forms of the winners 添加变异和配对后的胜出者
    while len(pop)<popsize:
      if random.random()<mutprob:

        # Mutation 变异
        c=random.randint(0,topelite)
        pop.append(mutate(ranked[c]))
      else:
      
        # Crossover 交叉
        c1=random.randint(0,topelite)
        c2=random.randint(0,topelite)
        pop.append(crossover(ranked[c1],ranked[c2]))
    
    # Print current best score 打印当前最优值
    print (f"成本值: {scores[0][0]}")
    
  return scores[0][1]
