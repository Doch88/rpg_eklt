#pragma once

#include <ros/ros.h>
#include <gflags/gflags.h>
#include <sensor_msgs/Image.h>
#include <dvs_msgs/EventArray.h>
#include <image_transport/image_transport.h>

#include <deque>
#include <csignal>
#include <mutex>
#include <fstream>
#include <string>
#include <functional>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <thread>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "patch.h"
#include "optimizer.h"
#include "viewer.h"

DECLARE_double(tracking_quality);
DECLARE_int32(vector_capacity);
DECLARE_int32(add_vector_capacity);


namespace tracker {
/**
 * @brief The Tracker class: uses Images to initialize corners and then tracks them using events.
 * Images are subscribed to and, when collected Harris Corners are extracted. Events falling in a patch around the corners
 * forming an event-frame-patch are used as observations and tracked versus the image gradient in the patch
 * at initialization.
 */
    class Tracker {
    public:
        Tracker(ros::NodeHandle &/*, viewer::Viewer &viewer*/);

        ~Tracker() {
            thread_running[0] = false;
            for (size_t i = 1; i < number_of_threads; i++) {
                threads[i].join();
                thread_running[i] = false;
            }
        }

    private:
        /**
       * @brief Initializes a viewer and optimizer with the first image. Also extracts first features.
       * @param image_it CV_8U gray-scale image on which features are extracted.
       */
        void init(const ImageBuffer::iterator &image_it);

        /**
         * @brief working thread
         */
        void processEvents();

        /**
         * @brief Blocks while there are no events in buffer
         * @param next event
        */
        inline void waitForEvent(dvs_msgs::Event &ev, int thread) {
            ros::Rate r(100);

            while (true) {
                {
                    std::unique_lock <std::mutex> lock(*events_mutex_[thread]);

                    if (threadIterators[thread] != events_.end()) {
                        auto tmpEv = *threadIterators[thread];
                        ev = tmpEv;
                        ++threadIterators[thread];
                        return;
                    }
                }
                r.sleep();
                VLOG(1) << "Thread " << thread << " - Waiting for events.";
            }
        }

        /**
         * @brief blocks until first image is received
         */
        void waitForFirstImage(ImageBuffer::iterator &current_image_it);

        /**
        * @brief Always assigns image to the first image before time  t_start
        */
        inline bool updateFirstImageBeforeTime(ros::Time t_start, ImageBuffer::iterator &current_image_it, int thread,
                                               OperationType op) {
            std::unique_lock <std::mutex> images_lock(images_mutex_);
            bool next_image = false;
            auto next_image_it = current_image_it;

            while (next_image_it->first < t_start) {
                ++next_image_it;
                if (next_image_it == images_.end())
                    break;

                if (next_image_it->first < t_start) {
                    next_image = true;
                    if (op == OperationType::UPDATE_IMAGE)
                        current_image_it = next_image_it;
                    break;
                }
            }

            return next_image;
        }

        /**
         * @brief checks all features if they can be bootstrapped
         */
        void bootstrapAllPossiblePatches(const ImageBuffer::iterator &image_it);

        /**
       * @brief bootstrapping features: Uses first two frames to initialize feature translation and optical flow.
       */
        void bootstrapFeatureKLT(Patch &patch, const cv::Mat &last_image, const cv::Mat &current_image, int thread,
                                 int index);

        /**
         * @brief bootstrapping features: Uses first event frame to solve for the best optical flow, given 0 translation.
         */
        void bootstrapFeatureEvents(Patch &patch, const cv::Mat &event_frame, int thread);

        /**
         * @brief add new features
         */
        void
        addFeatures(std::vector <std::vector<int>> &lost_indices, const ImageBuffer::iterator &image_it, int thread);

        /**
         * @brief update a patch with the new event
         */
        bool checkUpdatePatch(Patch &patch, const dvs_msgs::Event &event);

        void optimizePatch(Patch &patch, int thread, int index);

        /**
         * @brief reset patches that have been lost.
         */
        void resetPatches(Patches &new_patches, std::vector<int> &lost_indices, const ImageBuffer::iterator &image_it);

        /**
         * @brief initialize corners on an image
         */
        void initPatches(Patches &patches, std::vector<int> &lost_indices, const int &corners,
                         const ImageBuffer::iterator &image_it);

        /**
         * @brief extract patches
         */
        void
        extractPatches(Patches &patches, const int &num_patches, const ImageBuffer::iterator &image_it, int thread);

        inline void padBorders(const cv::Mat &in, cv::Mat &out, int p) {
            out = cv::Mat(in.rows + p * 2, in.cols + p * 2, in.depth());
            cv::Mat gray(out, cv::Rect(p, p, in.cols, in.rows));
            copyMakeBorder(in, out, p, p, p, p, cv::BORDER_CONSTANT);
        }

        /**
         * @brief checks if the optimization cost is above 1.6 (as described in the paper)
         */
        inline bool shouldDiscard(Patch &patch) {
            bool out_of_fov = (patch.center_.y < 0 || patch.center_.y >= sensor_size_.height || patch.center_.x < 0 ||
                               patch.center_.x >= sensor_size_.width);
            bool exceeded_error = patch.tracking_quality_ < FLAGS_tracking_quality;

            return exceeded_error || out_of_fov;
        }

        /**
         * @brief sets the number of events to process adaptively according to equation (15) in the paper
         */
        void setBatchSize(Patch &patch, const cv::Mat &I_x, const cv::Mat &I_y, const double &d);

        /**
         * @brief ros callbacks for images and events
         */
        void eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg);

        void imageCallback(const sensor_msgs::Image::ConstPtr &msg);

        /**
         * @brief Insert an event in the buffer while keeping the buffer sorted
         * This uses insertion sort as the events already come almost always sorted
         */
        inline void insertEventInSortedBuffer(const dvs_msgs::Event &e) {

            checkEventsCapacity();

            events_.push_back(e);

            // insertion sort to keep the buffer sorted
            // in practice, the events come almost always sorted,
            // so the number of iterations of this loop is almost always 0
            int j = (events_.size() - 1) - 1; // second to last element
            while (j >= 0 && events_[j].ts > e.ts) {
                events_[j + 1] = events_[j];
                j--;
            }
            events_[j + 1] = e;
        }

       /**
        * @brief Sort patches to have that the first ones are in the top left corner
        */
        void sortPatches(Patches &patches) {
            static const auto sort_function = [](const Patch &first, const Patch &second) -> bool {
                auto firstDistance = first.center_.x * first.center_.x + first.center_.y * first.center_.y;
                auto secondDistance = second.center_.x * second.center_.x + second.center_.y * second.center_.y;
                return firstDistance < secondDistance;
            };

            std::sort(patches.begin(), patches.end(), sort_function);
        }

       /**
        * @brief Distribute patches over all the threads.
        * This method assumes that the patches are sorted
        */
        void distributePatches(OperationType op = OperationType::NONE) {
            patch_per_thread.clear();

            for (int i = 0; i < number_of_threads; i++) {
                patch_per_thread.push_back(std::vector<Patch>());
            }

            // If the patches are sorted this part distribute patches in order to have
            // that nearby patches are assigned to different threads.
            // Doing this we have that if an area of the frame is particularly dense of features then
            // the relative computation will be distributed among all the threads.
            for (int i = 0; i < patches_.size(); i++) {
                int thread = i % number_of_threads;
                patch_per_thread[thread].push_back(patches_[i]);
            }

            // create lost indices using the lost_ attribute of the patches
            if (op == OperationType::INFER_LOST_INDICES) {
                for (int i = 0; i < lost_indices_.size(); i++) {
                    lost_indices_[i] = std::vector<int>();
                    for (int j = 0; j < patch_per_thread[i].size(); j++)
                        if (patch_per_thread[i][j].lost_)
                            lost_indices_[i].push_back(j);
                }
            } else {
                for (int i = 0; i < lost_indices_.size(); i++)
                    lost_indices_[i] = std::vector<int>();
            }
        }

      /**
       * @briefopposite operation of "distributePatches"
       * it merges all the threads' features in a single vector
       * the same operation we'll be done with the lost indices of every thread.
       */
       void mergeThread(std::vector <std::vector<int>> &lost_indices,
                        std::vector<int> &global_lost_indices, OperationType op = OperationType::NONE) {
           patches_.clear();

           for (int i = 0; i < number_of_threads; i++) {
               for (int j = 0; j < patch_per_thread[i].size(); j++)
                   patches_.push_back(patch_per_thread[i][j]);
           }

           // sort the vectors in order to have the lost features at the end.
           // doing this, we'll distribute the lost patches among all threads
           if (op == OperationType::SORT_BY_LOST_INDICES) {
               static const auto fun = [](const Patch &a, const Patch &b) {
                   if (!a.lost_ && b.lost_)
                       return true;
                   else return false;
               };

               std::sort(patches_.begin(), patches_.end(), fun);

               int i;
               for (i = 0; i < patches_.size(); i++) {
                   if (patches_[i].lost_)
                       break;
               }

               for (; i < patches_.size(); i++)
                   global_lost_indices.push_back(i);
           } else if (op == OperationType::INFER_LOST_INDICES){
               for (int i = 0; i < patches_.size(); i++) {
                   if (patches_[i].lost_)
                       global_lost_indices.push_back(i);
               }
           } else {
               for (int thr = 0; thr < lost_indices.size(); thr++)
                   for (int num : lost_indices[thr])
                       global_lost_indices.push_back(num + thr * patch_per_thread[thr].size());
           }
       }

       void cleanEvents() {
           std::vector <std::unique_lock<std::mutex>> mutexVector;
           for (int i = 0; i < events_mutex_.size(); i++)
               mutexVector.push_back(std::unique_lock<std::mutex>(*events_mutex_[i]));

           EventBuffer::iterator minIt = threadIterators[0];
           /*bool changed = false;
           for(auto && it: threadIterators) {
               if(std::distance(minIt, it) < 0) {
                   VLOG(1) << std::distance(minIt, it);
                   minIt = it;
                   changed = true;
               }
           }
           if(changed)*/
           events_.erase(events_.begin(), minIt);

           for(int i = 0; i < threadIterators.size(); i++) {
               threadIterators[i] = events_.begin();
           }

           mutexVector.clear();
       }

       void checkEventsCapacity() {
           if(events_.size()+1 < events_.capacity())
               return;

           std::vector <std::unique_lock<std::mutex>> mutexVector;
           for (int i = 0; i < events_mutex_.size(); i++)
               mutexVector.push_back(std::unique_lock<std::mutex>(*events_mutex_[i]));


           VLOG(1) << "Updating events capacity...";

           std::vector<int> distances;
           for(auto && it : threadIterators) {
               distances.push_back(std::distance(events_.begin(), it));
           }

           unsigned int addValue = 0;
           if(events_.capacity() + FLAGS_add_vector_capacity >= events_.max_size()) {
               addValue = events_.max_size() - events_.capacity() - 2;
               VLOG(1) << "Reached events max size!";
           } else
               addValue = FLAGS_add_vector_capacity;

           events_.reserve(events_.capacity() + addValue);

           VLOG(1) << "Reserved " << events_.capacity();

           for(int i = 0; i < threadIterators.size(); i++) {
               threadIterators[i] = events_.begin() + distances[i];
           }

           mutexVector.clear();
       }

       int number_of_threads;

       cv::Size sensor_size_;

       // image flags
       bool got_first_image_;

       // pointers to most recent image and time
       ImageBuffer::iterator current_image_it_;

       // buffers for images and events
       EventBuffer events_;

       std::vector<EventBuffer::iterator> threadIterators;

       ImageBuffer images_;

       // ros
       ros::Subscriber event_sub_;

       image_transport::Subscriber image_sub_;
       image_transport::ImageTransport it_;
       ros::NodeHandle nh_;

       // patch parameters
       Patches patches_;
       //std::map<int, std::pair<cv::Mat, cv::Mat>> patch_gradients_;
       std::vector <std::vector<int>> lost_indices_;
       std::vector <std::vector<Patch>> patch_per_thread;
       std::vector <ros::Time> most_current_time_;

       // delegation
       viewer::Viewer *viewer_ptr_ = nullptr;
       nlls::Optimizer optimizer_;

       // mutex
       std::vector <std::unique_ptr<std::mutex>> events_mutex_; // mutex used when events are added or used on the patches
       std::mutex images_mutex_; // mutex used when a new image arrives
       std::vector <std::unique_ptr<std::mutex>> distribute_mutex; // mutex used when patches are changed or distributed
       std::mutex file_mutex; // mutex used to synchronize writing on file
       std::mutex thread_sync_mutex; // mutex used to synchronize thread

       // threads
       std::vector <std::thread> threads; // vector containing all the threads
       std::vector<bool> entered_thread; // states if a certain thread entered the sync part of the algorithm
       std::vector<bool> thread_running; // states that a certain thread is running

       // tracks file
       std::ofstream tracks_file_;
   };

}
