library(GA)

# Set the number of people to N
n_people <- 100

# Define the fitness function that evaluates the number of queen collisions in a board state
fitness_function <- function(population) {
  n_people <- length(population)
  collisions <- 0
  for (i in 1:(n_people - 1)) {
    distance <- abs(population[i] - population[i + 1])
    if (distance == i) {
      print(distance)
      print(i)
      collisions <- collisions + 1
    }
  }
  return(collisions)
}

# Define the population sizes to be tested
pop_sizes <- c(50, 100, 200, 500, 1000, 2000)

# Initialize an empty vector to store the number of collisions for each population size
collisions <- numeric(length(pop_sizes))

# Set the maximum number of iterations for the GA algorithm
max_iter = 50

# Loop through each population size and run the GA algorithm
for (i in 1:length(pop_sizes)) {
  set.seed(1)
  GA <- ga(
    type = "permutation",
    maxiter = max_iter,
    fitness = fitness_function,
    lower = 1,
    upper = n_people,
    popSize = pop_sizes[i],
    pmutation = 0.5,
    pcrossover = 0.8
  )
  
  # Check if there is a solution
  if (nrow(GA@solution) > 0) {
    summary(GA)
    plot_pop = plot(GA)
    best_solution <- GA@solution[1, ]
    collisions[i] <- fitness_function(best_solution)
  } else {
    collisions[i] <- NA
  }
}

# Create a data frame to store the results of the experiment
results_table <-
  data.frame(Population_Size = pop_sizes, Min_Collisions = collisions)

print(results_table)
