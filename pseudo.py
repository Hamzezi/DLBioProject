X (K, D1) K = 5, D1 = 100 # class C, 3 classes

avg_concepts_1 = []

for k in range(len(x)):
    concepts = model(X[k]) # (N, D2) (10)
    avg_concepts_1.append(concepts)

avg_concepts_1 # (K, N, D2)

avg_concepts_1 = torch.mean(avg_concepts_1, dim=0)  # (N, D2), prototype for class C

x_new # (D1)

x_new_concepts = model(x_new) # (N, D2)

torch.euclidean_dist(x_new_concepts, avg_concepts_1) # for each n in N


c1 = torch.euclidean_dist(x_new_concepts, avg_concepts_1).sum(dim=0) # (D2)
c2= torch.euclidean_dist(x_new_concepts, avg_concepts_2)
c3= torch.euclidean_dist(x_new_concepts, avg_concepts_3)

torch.softmax([c1, c2, c3])
pred_class = torch.argmax([c1, c2, c3])