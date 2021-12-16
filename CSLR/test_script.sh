#!/bin/bash

for number in {1..30}
do
tzq config/$1 test --epoch $number
done
exit 0