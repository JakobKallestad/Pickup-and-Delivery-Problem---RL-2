from random import shuffle

sol = [1,2,3,4,5]

temp_sol = [97, 98, 99]

ind = 5


solution = sol[:ind] + temp_sol + sol[ind:]

print(solution)