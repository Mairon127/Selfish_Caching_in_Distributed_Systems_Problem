# **Selfish Caching in Distributed Systems: A Game-Theoretic Analysis - Implementation**

## **Overview**
This project implements a program inspired by the research presented in the paper  
**_"Selfish Caching in Distributed Systems: A Game-Theoretic Analysis"_**  
by Byung-Gon Chun, Kamalika Chaudhuri, Hoeteck Wee, Marco Barreno, Christos H. Papadimitriou, and John Kubiatowicz from the University of California, Berkeley.

The program explores resource replication in distributed networks, where selfish nodes independently decide whether to cache data based on cost minimization. The system is modeled using **game theory**, focusing on **Nash equilibria**, the **price of anarchy (PoA)**, and the **optimistic price of anarchy (OptPoA)**.

## **Features**
- 📌 **Distance Matrix Generation**: Creates a symmetric distance matrix for network nodes.
- 🌐 **Network Topology Simulation**: Supports different topologies like **complete graphs, grids, stars, buses, and random graphs**.
- 📊 **Global Replication Optimization**: Solves the **global caching problem** using **linear programming (PuLP solver)**.
- 🎯 **Heuristic-Based Solution**: Implements an alternative **game-theoretic heuristic approach**.
- ⚖️ **Nash Equilibrium Computation**: Identifies stable configurations where no node has an incentive to change its strategy.
- 🔍 **Price of Anarchy Evaluation**: Measures system efficiency loss due to selfish behavior.
- 🖼 **Graph Visualization**: Displays network topologies and replication configurations.

## **Methodology**
The implementation follows the **game-theoretic model** described in the referenced paper:

1. **Each node incurs two types of costs**:
   - **Replication Cost (α)**: Cost of storing a replica.
   - **Access Cost**: Distance cost for accessing a remote replica.

2. **Nodes behave selfishly**, selecting strategies to minimize individual costs.

3. **Nash equilibrium analysis**:
   - Determines if a given caching strategy is stable.
   - Explores variations of Nash equilibria for different network conditions.

4. **Performance Metrics**:
   - **Price of Anarchy (PoA)**: Ratio of the worst-case Nash equilibrium to the social optimum.
   - **Optimistic PoA (OptPoA)**: Ratio of the best Nash equilibrium to the social optimum.

## **Installation & Dependencies**
This project requires **Python 3.x** and the following libraries:

```bash
pip install numpy networkx matplotlib pulp
```

## **Usage**
To run the program, execute:

```bash
python test3jednamacierz.py
```
## **Reference Paper**
This work is based on the research presented in:
Selfish Caching in Distributed Systems: A Game-Theoretic Analysis
Byung-Gon Chun, Kamalika Chaudhuri, Hoeteck Wee, Marco Barreno, Christos H. Papadimitriou, John Kubiatowicz
University of California, Berkeley
