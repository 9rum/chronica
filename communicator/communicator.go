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

// The communicator package implements an intermediary to communicate with
// Chronica's scheduler.  The primitives are based on the syntax of the Message
// Passing Interface (MPI); the communicator runtime always starts with Init and
// ends with Finalize. At the beginning of each training epoch, Bcast is invoked
// to broadcast the schedule for the corresponding epoch to all workers.
package communicator

import (
	"context"
	"os"
	"os/signal"
	"runtime"
	"sync"
	"syscall"

	"github.com/9rum/chronica/internal/data"
	"github.com/9rum/chronica/scheduler"
	"github.com/golang/glog"
	"github.com/golang/protobuf/ptypes/empty"
)

// communicatorServer implements the server API for Communicator service.
type communicatorServer struct {
	UnimplementedCommunicatorServer
	scheduler scheduler.Scheduler
	done      chan<- os.Signal
	fanin     chan struct{}
	fanout    []chan []int
	steps     int
}

// NewCommunicatorServer creates a new communicator server with the given
// arguments.
func NewCommunicatorServer(done chan<- os.Signal, worldSize int) CommunicatorServer {
	fanout := make([]chan []int, 0, worldSize)
	for len(fanout) < cap(fanout) {
		fanout = append(fanout, make(chan []int))
	}
	return &communicatorServer{
		done:   done,
		fanin:  make(chan struct{}),
		fanout: fanout,
	}
}

// Init initializes the training environment.
func (c *communicatorServer) Init(ctx context.Context, in *InitRequest) (*empty.Empty, error) {
	go func() {
		c.fanin <- struct{}{}
	}()

	if in.GetRank() == 0 {
		go func() {
			for range c.fanout {
				<-c.fanin
			}
			c.init(len(c.fanout), int(in.GetBatchSize()), cast[int64, int](in.GetSizes()), cast[int64, int](in.GetGroups()), in.GetPartition(), in.GetType())
			for _, ch := range c.fanout {
				ch <- nil
			}
		}()
	}

	<-c.fanout[in.GetRank()]
	return new(empty.Empty), nil
}

// init initializes the data set and scheduler with the given arguments.
func (c *communicatorServer) init(worldSize, batchSize int, sizes, groups []int, partition bool, typ Schedule) {
	glog.Infof("Init called with world size: %d batch size: %d type: %s", worldSize, batchSize, typ)

	// initialize the data set and scheduler
	dataset := data.New(sizes, groups, partition)
	c.scheduler = scheduler.New(dataset, worldSize, batchSize, sizes, typ)
	c.steps = ceil(len(sizes), batchSize)
}

// cast casts the given slice.
func cast[T, U ~int | ~int64](slice []T) []U {
	out := make([]U, len(slice))
	stride := ceil(len(slice), runtime.NumCPU())

	var wg sync.WaitGroup
	for base := 0; base < len(slice); base += stride {
		wg.Add(1)
		go func(base int) {
			defer wg.Done()
			for index := base; index < min(base+stride, len(slice)); index++ {
				out[index] = U(slice[index])
			}
		}(base)
	}
	wg.Wait()

	return out
}

// ceil returns the least integer value greater than or equal to numerator / denominator.
// This is an alternative to the Ceil function in the standard math package.
func ceil(numerator, denominator int) int {
	if numerator%denominator == 0 {
		return numerator / denominator
	}
	return numerator/denominator + 1
}

func min(lhs, rhs int) int {
	if lhs < rhs {
		return lhs
	}
	return rhs
}

// Bcast broadcasts the schedule to all workers. If the scheduler provides a
// feedback-directed optimization, the performance indicators in the given
// feedback are used to estimate the training time.
func (c *communicatorServer) Bcast(ctx context.Context, in *BcastRequest) (*BcastResponse, error) {
	glog.Infof("epoch: %d Bcast called from rank %d", in.GetEpoch(), in.GetRank())

	go func() {
		c.scheduler.OnEpochEnd(in.GetEpoch(), in.GetRank(), in.GetCoefficient(), in.GetIntercept())
		c.fanin <- struct{}{}
	}()

	if in.GetRank() == 0 {
		go func() {
			for range c.fanout {
				<-c.fanin
			}
			for rank, indices := range scheduler.NextN(c.scheduler, c.steps) {
				c.fanout[rank] <- indices
			}
		}()
	}

	indices := <-c.fanout[in.GetRank()]
	return &BcastResponse{Indices: cast[int, int64](indices)}, nil
}

// Finalize terminates the training environment.
func (c *communicatorServer) Finalize(ctx context.Context, in *empty.Empty) (*empty.Empty, error) {
	glog.Info("Finalize called")
	defer glog.Flush()
	defer c.close()

	c.scheduler.OnTrainEnd()
	c.scheduler = nil

	return new(empty.Empty), nil
}

// close closes all open channels and notifies the main goroutine that the
// communicator runtime has ended.
func (c *communicatorServer) close() {
	close(c.fanin)
	for _, ch := range c.fanout {
		close(ch)
	}
	signal.Notify(c.done, syscall.SIGTERM)
	close(c.done)
}
