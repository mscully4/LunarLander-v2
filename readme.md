
I used Double-Deep-Q Learning to beat the Lunar Lander v2 game in OpenAi Gym.
My solution is done in Python and uses Keras/TensorFlow.

My model in action!

https://user-images.githubusercontent.com/36865337/132965823-09c54900-d63e-4ba9-aeab-1362c6b5a268.mp4

I played around with different hyperparameters to see how it would change the results

If you change alpha...
| UUID                                 |  Alpha  |  Gamma  |  Decay  | Best Score |
|:------------------------------------ |:-------:|:-------:|:-------:|:----------:|
| a65452ae-d636-11eb-85cb-02252fa97aca | 0.00005 | 0.99    | 0.995   |   244.04   |
| 4662f408-d5f6-11eb-984f-02252fa97aca | 0.0001  | 0.99    | 0.995   |   269.98   |
| 25c643ee-d9ac-11eb-8997-02362b9e3ec8 | 0.0002  | 0.99    | 0.995   |   261.40   |
| b4337008-d627-11eb-9e60-02252fa97aca | 0.0002  | 0.99    | 0.995   |   211.35   |
| 58db58e2-d618-11eb-ae32-02252fa97aca | 0.0003  | 0.99    | 0.995   |   220.77   | 

If you change gamma...
| UUID                                 |  Alpha  |  Gamma  |  Decay  | Best Score |
|:------------------------------------ |:-------:|:-------:|:-------:|:----------:|
| c042ccce-d6ed-11eb-aed4-0267ce578a34 | 0.0001  | 0.95    | 0.995   |   -70.1    |
| 75b7f3e6-d6fc-11eb-9908-0267ce578a34 | 0.0001  | 0.97    | 0.995   |   244.03   |
| 4662f408-d5f6-11eb-984f-02252fa97aca | 0.0001  | 0.99    | 0.995   |   269.98   |
| d5504312-da03-11eb-a01c-02362b9e3ec8 | 0.0001  | 0.995   | 0.995   |   269.04   |
| 80928d7c-d77d-11eb-b311-0a111e13e966 | 0.0001  | 0.999   | 0.995   |   264.11   |

If you change the rate of epsilon decay...
| UUID                                 |  Alpha  |  Gamma  |  Decay  | Best Score |
|:------------------------------------ |:-------:|:-------:|:-------:|:----------:|
| 9e73fbac-d906-11eb-951a-02362b9e3ec8 | 0.0001  | 0.99    | 0.991   |   257.59   |
| 82ba7a18-d87e-11eb-a531-02362b9e3ec8 | 0.0001  | 0.99    | 0.993   |   278.39   |
| 4662f408-d5f6-11eb-984f-02252fa97aca | 0.0001  | 0.99    | 0.995   |   269.98   |
| bb38c0be-d867-11eb-8124-02362b9e3ec8 | 0.0001  | 0.99    | 0.997   |   221.19   |
