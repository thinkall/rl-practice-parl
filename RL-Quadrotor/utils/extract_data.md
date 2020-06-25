I forgot to record test scores during the training process, so I extracted the data from the logs.

```shell
cat quadrotor-velocity.ipynb | grep -n 'Test reward: .*, Pid 2091' | sed -e 's/^.*Test reward: //' -e 's/, Pid 2091\\n"//' -e 's/\(....\)\(..\)\(..\)/\1\2\3/'

cat quadrotor-velocity.ipynb | grep -n 'Steps .*, Test reward: .*, Pid 2091' | grep -n 'Steps .*, Test' | sed -e 's/^.*Steps //' -e 's/, Test$",//' -e 's/\(....\)\(..\)\(..\)/\1\2\3/'
```
