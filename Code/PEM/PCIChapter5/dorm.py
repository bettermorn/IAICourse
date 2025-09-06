import random
import math

# The dorms, each of which has two available spaces 每个宿舍有两个隔间
dorms=['Athena','Bacchus','Hercules','Pluto','Zeus']

# People, along with their first and second choices 每个学生的两个选择
prefs=[('Toby', ('Bacchus', 'Hercules')),
       ('Steve', ('Zeus', 'Pluto')),
       ('Karen', ('Athena', 'Zeus')),
       ('Sarah', ('Zeus', 'Pluto')),
       ('Dave', ('Athena', 'Bacchus')), 
       ('Jeff', ('Hercules', 'Pluto')), 
       ('Fred', ('Pluto', 'Athena')), 
       ('Suzie', ('Bacchus', 'Hercules')), 
       ('Laura', ('Bacchus', 'Hercules')), 
       ('James', ('Hercules', 'Athena'))]

# [(0,9),(0,8),(0,7),(0,6),...,(0,0)] 将每个学生依序安置于各空槽（0-9，5个宿舍）内，第一位可置于10个槽中的任何一个，第二位可置于剩余9个槽中的任何一个，依次类推。
# 搜索的定义域满足这一约束
domain=[(0,(len(dorms)*2)-i-1) for i in range(0,len(dorms)*2)]


# 打印宿舍分配结果
def printsolution(vec):
  # vec 必须满足第一项的值介于0-9，第二项的值介于0-8，第二项的值介于0-7等，否则会抛出异常。  
  print(f"宿舍分配序列：{vec}")
  slots = []
  realvec = []  
  # Create two slots for each dorm 为每个宿舍建两个槽
  for i in range(len(dorms)): 
    slots += [i,i]

  # Loop over each students assignment 遍历每一名学生的安置情况
  for i in range(len(vec)):
    x = int(vec[i])

    # Choose the slot from the remaining ones 从剩余槽中选择
    dorm = dorms[slots[x]]
    realvec.append(dorms.index(dorm))  
    # Show the student and assigned dorm 输出学生及其被分配的宿舍
    print (f"{prefs[i][0]}-首选{prefs[i][1][0]}-次选{prefs[i][1][1]}-实际分配-{dorm}")
      
    # Remove this slot 删除该槽
    del slots[x]

  print(f"实际分配结果是：{realvec}")
  realcost = dormrealcost(realvec)  
  print(f"实际分配的成本值是：{realcost}") 

# 成本函数
def dormrealcost(vec):
  if vec is None:
    return 10000  # 返回一个很高的成本值

  cost = 0

  # Loop over each student 遍历每一名学生
  for i in range(len(vec)):
    x = int(vec[i])
    dorm = dorms[x]
    pref = prefs[i][1]
    # First choice costs 0, second choice costs 1 首选成本值为0，次选成本值为1
    if pref[0] == dorm: 
        cost += 0
    elif pref[1] == dorm: 
        cost += 1
    else: 
        cost += 3
    # Not on the list costs 3 不在选择之列则成本值为3

    
  return cost



# 成本函数
def dormcost(vec):
  if vec is None:
    return 10000  # 返回一个很高的成本值
    
  cost = 0
  # Create list a of slots 每个宿舍建两个槽 0-4代表宿舍编号 0-9代表分配给学生的隔间编号
  slots = [0,0,1,1,2,2,3,3,4,4]

  # Loop over each student 遍历每一名学生
  for i in range(len(vec)):
    x = int(vec[i])
    dorm = dorms[slots[x]]
    pref = prefs[i][1]
    # First choice costs 0, second choice costs 1 首选成本值为0，次选成本值为1
    if pref[0] == dorm: 
        cost += 0
    elif pref[1] == dorm: 
        cost += 1
    else: 
        cost += 3
    # Not on the list costs 3 不在选择之列则成本值为3

    # Remove selected slot 删除选中的槽
    del slots[x]
    
  return cost
