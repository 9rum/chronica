// Copyright 2022 Sogang University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package data

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/9rum/chronica/internal/btree"
)

func init() {
	seed := time.Now().Unix()
	fmt.Println(seed)
	rand.Seed(seed)
}

func TestShardedDataset(t *testing.T) {
	const (
		datasetSize = 10000
		batchSize   = 10
	)
	sizes := rand.Perm(datasetSize)
	dataset := NewShardedDataset[*btree.ItemBase](sizes)

	for epoch := 0; epoch < 10; epoch++ {
		for step := 0; step < datasetSize/batchSize; step++ {
			for index := step * batchSize; index < (step+1)*batchSize; index++ {
				if _, size := dataset.Getitem(sizes[index], sizes[index]); size != sizes[index] {
					t.Fatalf("did not find %d", sizes[index])
				}
			}
		}
		dataset.OnEpochEnd()
	}
	dataset.OnTrainEnd()

	for size := range sizes {
		sizes[size] = size
	}
	dataset = NewShardedDataset[*btree.ItemBase](sizes)

	for epoch := 0; epoch < 10; epoch++ {
		for step := 0; step < datasetSize/batchSize; step++ {
			for index := step * batchSize; index < (step+1)*batchSize; index++ {
				if _, size := dataset.Getitem(sizes[index], sizes[index]); size != sizes[index] {
					t.Fatalf("did not find %d", sizes[index])
				}
			}
		}
		dataset.OnEpochEnd()
	}
	dataset.OnTrainEnd()
}

func TestPartitionedDataset(t *testing.T) {
	const (
		datasetSize = 10000
		batchSize   = 40
		worldSize   = 4
	)
	sizes := make([][]int, worldSize)
	for rank := range sizes {
		sizes[rank] = rand.Perm(datasetSize / worldSize)
	}
	dataset := NewPartitionedDataset[*btree.ItemBase](sizes)

	for epoch := 0; epoch < 10; epoch++ {
		for step := 0; step < datasetSize/batchSize; step++ {
			for rank := 0; rank < worldSize; rank++ {
				for index := step * batchSize / worldSize; index < (step+1)*batchSize/worldSize; index++ {
					if _, size := dataset.Getitem(rank, sizes[rank][index]); size != sizes[rank][index] {
						t.Fatalf("did not find %d", sizes[rank][index])
					}
				}
			}
		}
		dataset.OnEpochEnd()
	}
	dataset.OnTrainEnd()

	for _, partition := range sizes {
		for size := range partition {
			partition[size] = size
		}
	}
	dataset = NewPartitionedDataset[*btree.ItemBase](sizes)

	for epoch := 0; epoch < 10; epoch++ {
		for step := 0; step < datasetSize/batchSize; step++ {
			for rank := 0; rank < worldSize; rank++ {
				for index := step * batchSize / worldSize; index < (step+1)*batchSize/worldSize; index++ {
					if _, size := dataset.Getitem(rank, sizes[rank][index]); size != sizes[rank][index] {
						t.Fatalf("did not find %d", sizes[rank][index])
					}
				}
			}
		}
		dataset.OnEpochEnd()
	}
	dataset.OnTrainEnd()
}

const benchmarkDatasetSize = 10000

func BenchmarkShardedDataset(b *testing.B) {
	b.StopTimer()
	const batchSize = 10
	sizes := rand.Perm(benchmarkDatasetSize)
	b.StartTimer()
	dataset := NewShardedDataset[*btree.ItemBase](sizes)

	for epoch := 0; epoch < b.N; epoch++ {
		for step := 0; step < benchmarkDatasetSize/batchSize; step++ {
			for index := step * batchSize; index < (step+1)*batchSize; index++ {
				dataset.Getitem(sizes[index], sizes[index])
			}
		}
		dataset.OnEpochEnd()
	}
	dataset.OnTrainEnd()
}

func BenchmarkPartitionedDataset(b *testing.B) {
	b.StopTimer()
	const (
		batchSize = 40
		worldSize = 4
	)
	sizes := make([][]int, worldSize)
	for rank := range sizes {
		sizes[rank] = rand.Perm(benchmarkDatasetSize / worldSize)
	}
	b.StartTimer()
	dataset := NewPartitionedDataset[*btree.ItemBase](sizes)

	for epoch := 0; epoch < b.N; epoch++ {
		for step := 0; step < benchmarkDatasetSize/batchSize; step++ {
			for rank := 0; rank < worldSize; rank++ {
				for index := step * batchSize / worldSize; index < (step+1)*batchSize/worldSize; index++ {
					dataset.Getitem(rank, sizes[rank][index])
				}
			}
		}
		dataset.OnEpochEnd()
	}
	dataset.OnTrainEnd()
}
