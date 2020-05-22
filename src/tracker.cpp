#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/video/tracking.hpp>

#include "tracker.h"

DECLARE_int32(patch_size);
DECLARE_int32(batch_size);
DECLARE_int32(min_distance);
DECLARE_int32(lk_window_size);
DECLARE_int32(num_pyramidal_layers);
DECLARE_int32(max_corners);
DECLARE_int32(min_corners);
DECLARE_int32(update_every_n_events);
DECLARE_int32(block_size);
DECLARE_int32(thread_number);

DECLARE_bool(display_features);

DECLARE_string(tracks_file_txt);
DECLARE_string(bootstrap);

DECLARE_double(displacement_px);
DECLARE_double(first_image_t);
DECLARE_double(quality_level);
DECLARE_double(k);


namespace tracker {

    Tracker::Tracker(ros::NodeHandle &nh/*, viewer::Viewer &viewer*/)
            : nh_(nh), got_first_image_(false), sensor_size_(0, 0),/* viewer_ptr_(&viewer),*/ it_(nh) {
        event_sub_ = nh_.subscribe("events", 10, &Tracker::eventsCallback, this);
        image_sub_ = it_.subscribe("images", 1, &Tracker::imageCallback, this);

        std::thread eventProcessingThread(&Tracker::processEvents, this);
        eventProcessingThread.detach();

        number_of_threads = FLAGS_thread_number;

        // initialize all the structures that will be used for the threads
        for (int i = 0; i < number_of_threads; i++) {
            events_.emplace_back();
            events_mutex_.push_back(std::make_unique<std::mutex>());
            distribute_mutex.push_back(std::make_unique<std::mutex>());
            lost_indices_.push_back(std::vector<int>());
            entered_thread.emplace_back(false);
            thread_running.emplace_back(true);
        }
    }

    void Tracker::waitForFirstImage(ImageBuffer::iterator &current_image_it) {
        ros::Rate r(30);
        while (!got_first_image_) {
            r.sleep();

            VLOG_EVERY_N(1, 30) << "Waiting for first image.";

            if (images_.empty())
                continue;

            VLOG(1) << "Found first image.";

            current_image_it = images_.begin();

            got_first_image_ = true;
        }
    }

    void Tracker::initPatches(Patches &patches, std::vector<int> &lost_indices, const int &corners,
                              const ImageBuffer::iterator &image_it) {
        // extract Harris corners
        extractPatches(patches, corners, image_it, 0);

        sortPatches(patches);

        // fill up patches to full capacity and set all of the filled patches to lost
        for (int i = patches.size(); i < corners; i++) {
            patches.emplace_back(true);
            lost_indices.push_back(i);
        }

        // extract log image gradient patches that act as features and are used to
        // compute the adaptive batchSize as per equation 15
        const int &p = (FLAGS_patch_size - 1) / 2;
        cv::Mat I_x, I_y, I_x_padded, I_y_padded;
        optimizer_.getLogGradients(image_it->second, I_x, I_y);

        padBorders(I_x, I_x_padded, p);
        padBorders(I_y, I_y_padded, p);

        for (int i = 0; i < patches.size(); i++) {
            Patch &patch = patches[i];
            if (patch.lost_) {
                patch.gradients = std::make_pair(cv::Mat::zeros(2 * p + 1, 2 * p + 1, CV_64F),
                                                 cv::Mat::zeros(2 * p + 1, 2 * p + 1, CV_64F));
            } else {
                const int x_min = patch.center_.x;
                const int y_min = patch.center_.y;

                patch.gradients = std::make_pair(
                        I_x_padded.rowRange(y_min, y_min + 2 * p + 1).colRange(x_min, x_min + 2 * p + 1).clone(),
                        I_y_padded.rowRange(y_min, y_min + 2 * p + 1).colRange(x_min, x_min + 2 * p + 1).clone());

                // sets adaptive batch size based on gradients according to equation (15) in the paper
                setBatchSize(patch, patch.gradients.first, patch.gradients.second, FLAGS_displacement_px);
            }
        }
    }

    void Tracker::init(const ImageBuffer::iterator &image_it) {
        // starts a file where feature track updates are recorded on each row (feature id, t, x, y)
        if (FLAGS_tracks_file_txt != "") {
            VLOG(1) << "Will write tracks data to '" << FLAGS_tracks_file_txt << "'.";
            tracks_file_.open(FLAGS_tracks_file_txt);
        }

        // extracts corners and extracts patches around them.
        // we'll need only the first element of lost_indices_ as this initialization is done before thread detachments
        initPatches(patches_, lost_indices_[0], FLAGS_max_corners, image_it);

        // initializes the image gradients in x and y directions for the first image
        // and initializes the ceres cubic interpolator for use in the optimizer
        optimizer_.precomputeLogImageArray(patches_, image_it);

        // init the data arrays that are used by the viewer
        // the time stamp drawn on the viewer is zeroed at image_it->first
        if (FLAGS_display_features)
            viewer_ptr_->initViewData(image_it->first);
    }

    void Tracker::processEvents() {
        // blocks until first image arrives and sets current_image_it_ to first arrived image
        waitForFirstImage(current_image_it_);

        // initializes patches and viewer
        init(current_image_it_);

        int viewer_counter = 0;
        VLOG(1) << "Thread number:" << number_of_threads;

        // we'll need to distribute the patches among all the threads in order to have a right load for every thread
        distributePatches();

        static double endTime = 1468941042.740955591;

        // let's create some timers that will be used for calculating the performance of the algorithm
        ros::Time beginTime;
        beginTime.fromSec(endTime);
        std::chrono::time_point <std::chrono::system_clock> begin = std::chrono::system_clock::now();
        ros::Time init = current_image_it_->first;

        for (int i = 0; i < number_of_threads; i++) {
            most_current_time_.push_back(current_image_it_->first);
            VLOG(3) << "Timestamp of thread " << i << ":" << std::to_string(current_image_it_->first.toSec());

            threads.push_back(std::thread([&, i]() {
                dvs_msgs::Event ev;

                int prev_num_features_tracked = 0;
                std::chrono::time_point <std::chrono::system_clock> start = std::chrono::system_clock::now();

                // main loop for every thread
                while (thread_running[i]) {
                    // blocks until first event is found
                    waitForEvent(ev, i);

                    const cv::Point2f point(ev.x, ev.y);

                    // go through each patch and update the event frame with the new event
                    int num_features_tracked = 0;

                    {
                        // critical section of the thread, we'll need to avoid that
                        // the patches change within this part
                        std::unique_lock <std::mutex> lock(*distribute_mutex[i]);

                        num_features_tracked = patch_per_thread[i].size();

                        for (size_t j = 0; j < patch_per_thread[i].size(); j += 1) {
                            if (checkUpdatePatch(patch_per_thread[i][j], ev)) {
                                optimizePatch(patch_per_thread[i][j], i, j);
                            }

                            // count tracked features
                            if (patch_per_thread[i][j].lost_)
                                num_features_tracked--;
                        }
                    }

                    // keep track of the most current time with latest time stamp from event
                    if (ev.ts >= most_current_time_[i])
                        most_current_time_[i] = ev.ts;

                    // check if this thread has reached the timestamp of a new image
                    if (updateFirstImageBeforeTime(most_current_time_[i], current_image_it_, i, OperationType::NONE)) {
                        // this thread is ready to be synchronized
                        entered_thread[i] = true;

                        // the first thread to pass over this mutex will lock other threads
                        std::unique_lock <std::mutex> lock(thread_sync_mutex);

                        // if this is the first thread to reach the new image, then it will wait (actively) for the other threads
                        while (updateFirstImageBeforeTime(most_current_time_[i], current_image_it_, i,
                                                          OperationType::NONE)) {
                            // check if all the threads are ready to be synchronized
                            bool allIn = true;
                            for (int i = 0; i < number_of_threads; i++)
                                if (!entered_thread[i])
                                    allIn = false;

                            if (!allIn)
                                continue;

                            ros::Time min = most_current_time_[0];

                            for(auto && t : most_current_time_) {
                                if(t < min)
                                    min = t;
                            }

                            if (updateFirstImageBeforeTime(min, current_image_it_, i,
                                                           OperationType::UPDATE_IMAGE)) // enter if new image found
                            {
                                VLOG(2) << "Thread " << i << " updated new image.";

                                // we'll compare time passed for the video with the time used for the tracking
                                if ((current_image_it_->first - init).toSec() >= 1.0) {
                                    VLOG(1) << "Time passed: " << most_current_time_[i] - init;
                                    std::chrono::time_point <std::chrono::system_clock> end = std::chrono::system_clock::now();
                                    double duration = std::chrono::duration<double>(end - start).count();
                                    VLOG(1) << "Time used for calculation: " << duration;
                                    init = current_image_it_->first;
                                    start = std::chrono::system_clock::now();
                                }

                                if(current_image_it_->first > beginTime) {
                                    std::chrono::time_point <std::chrono::system_clock> end = std::chrono::system_clock::now();
                                    double duration = std::chrono::duration<double>(end - begin).count();
                                    VLOG(1) << "=====>  Time total: " << duration << std::endl;
                                    raise(SIGKILL);
                                }

                                // bootstrap patches that need to be due to new image
                                if (FLAGS_bootstrap == "klt") {
                                    bootstrapAllPossiblePatches(current_image_it_);
                                }

                                // calculate the overall quantity of lost indices
                                int sum = 0;
                                for (auto vec : lost_indices_) {
                                    sum += vec.size();
                                }

                                // replenish features if there are too few
                                if (sum > FLAGS_max_corners - FLAGS_min_corners) {
                                    addFeatures(lost_indices_, current_image_it_, i);
                                } else {
                                    // if there are enough features we'll only redistribute them to all the threads
                                    // in order to rebalance to overall load
                                    std::vector <std::unique_lock<std::mutex>> mutexVector;
                                    for (int i = 0; i < distribute_mutex.size(); i++)
                                        mutexVector.push_back(std::unique_lock<std::mutex>(*distribute_mutex[i]));
                                    for (int i = 0; i < events_mutex_.size(); i++)
                                        mutexVector.push_back(std::unique_lock<std::mutex>(*events_mutex_[i]));

                                    std::vector<int> global_lost_indices;
                                    mergeThread(lost_indices_, global_lost_indices,
                                                OperationType::SORT_BY_LOST_INDICES);
                                    distributePatches(OperationType::INFER_LOST_INDICES);
                                }

                                // erase old image
                                auto image_it = current_image_it_;
                                image_it--;
                                images_.erase(image_it);

                                for (int i = 0; i < number_of_threads; i++)
                                    entered_thread[i] = false;

                                break;
                            }
                        }
                    }

                    if (prev_num_features_tracked > num_features_tracked) {
                        VLOG(2) << "Thread: " << i << " - Tracking " << num_features_tracked << " features.";
                    }

                    prev_num_features_tracked = num_features_tracked;

                    // update data for viewer
                    /*if (FLAGS_display_features && \
                        ++viewer_counter % FLAGS_update_every_n_events == 0)
                        viewer_ptr_->setViewData(patch_per_thread[i], most_current_time_, current_image_it_);*/
                }
            }));
        }

        threads[0].join();
    }

    bool Tracker::checkUpdatePatch(Patch &patch, const dvs_msgs::Event &event) {
        // if patch is lost or event does not fall within patch
        // or the event has occurred before the most recent patch timestamp
        // or the patch has not been bootstrapped yet, do not process event
        if (patch.lost_ ||
            (FLAGS_bootstrap == "klt" && !patch.initialized_) ||
            !patch.contains(event.x, event.y) ||
            patch.t_curr_ > event.ts)
            return false;

        patch.insert(event);

        // start optimization if there are update_rate new events in the patch
        int update_rate = std::min<int>(patch.update_rate_, patch.batch_size_);
        if (patch.event_buffer_.size() < patch.batch_size_ || patch.event_counter_ < update_rate)
            return false;
        return true;
    }

    void Tracker::optimizePatch(Patch &patch, int thread, int index) {
        // compute event frame according to equation (2) in the paper
        cv::Mat event_frame;
        patch.getEventFramesAndReset(event_frame);

        // bootstrap using the events
        ros::Time t_prev = patch.t_curr_;
        if (!patch.initialized_ && FLAGS_bootstrap == "events")
            bootstrapFeatureEvents(patch, event_frame, thread);

        // update feature position and recompute the adaptive batchsize
        optimizer_.optimizeParameters(event_frame, patch);

        if (tracks_file_.is_open()) {
            std::unique_lock <std::mutex> lock(file_mutex);
            tracks_file_ << patch.id_ << " " << patch.t_curr_ << " " << patch.center_.x << " " << patch.center_.y << " "
                         <<
                         std::to_string(thread) << std::endl;
        }

        setBatchSize(patch, patch.gradients.first, patch.gradients.second, FLAGS_displacement_px);

        if (!shouldDiscard(patch))
            return;

        // if the patch has been lost record it in lost_indices_
        patch.lost_ = true;
        lost_indices_[thread].push_back(index);
    }

    void Tracker::addFeatures(std::vector <std::vector<int>> &lost_indices, const ImageBuffer::iterator &image_it,
                              int thread) {
        // These mutex will prevent concurrent changes of the patches vectors
        std::vector <std::unique_lock<std::mutex>> mutexVector;
        for (int i = 0; i < distribute_mutex.size(); i++)
            mutexVector.push_back(std::unique_lock<std::mutex>(*distribute_mutex[i]));
        for (int i = 0; i < events_mutex_.size(); i++)
            mutexVector.push_back(std::unique_lock<std::mutex>(*events_mutex_[i]));

        // in this part we need to have all the patches in a single vector in order to replenish lost features, so
        // we merge every thread patch vector
        std::vector<int> global_lost_indices;
        mergeThread(lost_indices, global_lost_indices, OperationType::INFER_LOST_INDICES);

        // find new patches to replace them lost features
        Patches patches;
        extractPatches(patches, global_lost_indices.size(), image_it, thread);

        if (patches.size() != 0) {
            // pass the new image to the optimizer to use for future optimizations
            optimizer_.precomputeLogImageArray(patches, image_it);

            // reset all lost features with newly initialized ones
            resetPatches(patches, global_lost_indices, image_it);

            sortPatches(patches_);
            distributePatches();
        }

        mutexVector.clear();
    }

    void Tracker::bootstrapAllPossiblePatches(const ImageBuffer::iterator &image_it) {
        std::vector <std::unique_lock<std::mutex>> mutexVector;
        for (int i = 0; i < distribute_mutex.size(); i++)
            mutexVector.push_back(std::unique_lock<std::mutex>(*distribute_mutex[i]));

        for (int j = 0; j < number_of_threads; j++) {
            for (int i = 0; i < patch_per_thread[j].size(); i++) {
                Patch &patch = patch_per_thread[j][i];

                // if a patch is already bootstrapped, lost or has just been extracted
                if (patch.initialized_ || patch.lost_ || patch.t_init_ == image_it->first)
                    continue;

                // perform bootstrapping using KLT and the first 2 frames, and compute the adaptive batch size
                // with the newly found parameters
                bootstrapFeatureKLT(patch, images_[patch.t_init_], image_it->second, j, i);
                setBatchSize(patch, patch.gradients.first, patch.gradients.second,
                             FLAGS_displacement_px);
            }
        }

        mutexVector.clear();
    }

    void Tracker::setBatchSize(Patch &patch, const cv::Mat &I_x, const cv::Mat &I_y, const double &d) {
        // implements the equation (15) of the paper
        cv::Mat gradient = d * std::cos(patch.flow_angle_) * I_x + d * std::sin(patch.flow_angle_) * I_y;
        patch.batch_size_ = std::min<double>(cv::norm(gradient, cv::NORM_L1), FLAGS_batch_size);
        patch.batch_size_ = std::max<int>(5, patch.batch_size_);
    }

    void
    Tracker::resetPatches(Patches &new_patches, std::vector<int> &lost_indices, const ImageBuffer::iterator &image_it) {
        const int &p = (FLAGS_patch_size - 1) / 2;
        cv::Mat I_x, I_y, I_x_padded, I_y_padded;

        optimizer_.getLogGradients(image_it->second, I_x, I_y);
        padBorders(I_x, I_x_padded, p);
        padBorders(I_y, I_y_padded, p);

        for (int i = new_patches.size() - 1; i >= 0; i--) {
            int index = lost_indices[i];
            // for each lost feature decrement the ref counter of the optimizer (will free image gradients when no more
            // features use the image with timestamp patches_[index].t_init_
            optimizer_.decrementCounter(patches_[index].t_init_);

            // reset lost patches with new ones
            Patch &reset_patch = new_patches[i];
            patches_[index].reset(reset_patch.center_, reset_patch.t_init_);

            // reinitialize the image gradients of new features
            const int x_min = reset_patch.center_.x - p;
            const int y_min = reset_patch.center_.y - p;
            cv::Mat p_I_x = I_x_padded.rowRange(y_min, y_min + 2 * p + 1).colRange(x_min, x_min + 2 * p + 1);
            cv::Mat p_I_y = I_y_padded.rowRange(y_min, y_min + 2 * p + 1).colRange(x_min, x_min + 2 * p + 1);
            patches_[index].gradients = std::make_pair(p_I_x.clone(), p_I_y.clone());
            setBatchSize(reset_patch, patches_[index].gradients.first, patches_[index].gradients.second,
                         FLAGS_displacement_px);
        }
    }

    void Tracker::extractPatches(Patches &patches, const int &num_patches, const ImageBuffer::iterator &image_it,
                                 int thread) {
        std::vector <cv::Point2d> features;

        // mask areas which are within a distance min_distance of other features or along the border.
        int hp = (FLAGS_patch_size - 1) / 2;
        int h = sensor_size_.height;
        int w = sensor_size_.width;
        cv::Mat mask = cv::Mat::ones(sensor_size_, CV_8UC1);
        mask.rowRange(0, hp).colRange(0, w - 1).setTo(0);
        mask.rowRange(h - hp, h - 1).colRange(0, w - 1).setTo(0);
        mask.rowRange(0, h - 1).colRange(0, hp).setTo(0);
        mask.rowRange(0, h - 1).colRange(w - hp, w - 1).setTo(0);

        const int &min_distance = FLAGS_min_distance;
        for (Patch &patch: patches_) {
            if (patch.lost_) continue;

            double min_x = std::fmax(patch.center_.x - min_distance, 0);
            double max_x = std::fmin(patch.center_.x + min_distance, w - 1);
            double min_y = std::fmax(patch.center_.y - min_distance, 0);
            double max_y = std::fmin(patch.center_.y + min_distance, h - 1);
            mask.rowRange(min_y, max_y).colRange(min_x, max_x).setTo(0);
        }

        // extract harris corners which are suitable
        // since they correspond to strong edges which also generate alot of events.
        VLOG(3) << "Harris corner detector with N=" << num_patches << " quality=" << FLAGS_quality_level
                << " min_dist=" << FLAGS_min_distance << " block_size=" << FLAGS_block_size << " k=" << FLAGS_k
                << " image_depth=" << image_it->second.depth() << " mask_ratio="
                << cv::sum(mask)[0] / (mask.cols * mask.rows);

        cv::goodFeaturesToTrack(image_it->second, features, num_patches,
                                FLAGS_quality_level,
                                FLAGS_min_distance, mask,
                                FLAGS_block_size,
                                true,
                                FLAGS_k);

        // initialize patches centered at the features with an initial pixel warp
        VLOG(2) << "Extracted " << features.size() << " new features on image at t=" << std::setprecision(15)
                << image_it->first.toSec() << " s.";
        for (int i = 0; i < features.size(); i++) {
            const cv::Point2f &feature = features[i];
            patches.emplace_back(feature, image_it->first);
            Patch &patch = patches[patches.size() - 1];

            if (tracks_file_.is_open()) {
                std::unique_lock <std::mutex> lock(file_mutex);
                tracks_file_ << patch.id_ << " " << patch.t_curr_ << " " << patch.center_.x << " " <<
                             patch.center_.y << " -" << thread << std::endl;
            }
        }
    }

    void Tracker::bootstrapFeatureKLT(Patch &patch, const cv::Mat &last_image, const cv::Mat &current_image, int thread,
                                      int index) {
        // bootstrap feature by initializing its warp and optical flow with KLT on successive images
        std::vector <cv::Point2f> points = {patch.init_center_};
        std::vector <cv::Point2f> next_points;

        // track feature for one frame
        std::vector<float> error;
        std::vector <uchar> status;
        cv::Size window(FLAGS_lk_window_size, FLAGS_lk_window_size);
        cv::calcOpticalFlowPyrLK(last_image, current_image, points, next_points, status, error, window,
                                 FLAGS_num_pyramidal_layers);

        // compute optical flow angle as direction where the feature moved
        double opt_flow_angle = std::atan2(next_points[0].y - points[0].y, next_points[0].x - points[0].x);
        patch.flow_angle_ = opt_flow_angle;

        // initialize warping as pure translation to new point
        patch.warping_.at<double>(0, 2) = -(next_points[0].x - points[0].x);
        patch.warping_.at<double>(1, 2) = -(next_points[0].y - points[0].y);
        patch.warpPixel(patch.init_center_, patch.center_);

        // check if new patch has been lost due to leaving the fov
        bool should_discard = bool(
                patch.center_.y < 0 || patch.center_.y >= sensor_size_.height || patch.center_.x < 0 ||
                patch.center_.x >= sensor_size_.width);
        if (should_discard) {
            patch.lost_ = true;
            lost_indices_[thread].push_back(index);
        } else {
            patch.initialized_ = true;
            patch.t_curr_ = current_image_it_->first;
            std::unique_lock <std::mutex> lock(file_mutex);
            if (tracks_file_.is_open())
                tracks_file_ << patch.id_ << " " << patch.t_curr_ << " " << patch.center_.x << " " << patch.center_.y
                             << " "
                             << thread << std::endl;
        }
    }

    void Tracker::bootstrapFeatureEvents(Patch &patch, const cv::Mat &event_frame, int thread) {
        // Implement a bootstrapping mechanism for computing the optical flow direction via
        // \nabla I \cdot v= - \Delta E --> v = - \nabla I ^ \dagger \Delta E (assuming no translation or rotation
        // of the feature.
        cv::Mat &I_x = patch.gradients.first;
        cv::Mat &I_y = patch.gradients.second;

        double s_I_xx = cv::sum(I_x.mul(I_x))[0];
        double s_I_yy = cv::sum(I_y.mul(I_y))[0];
        double s_I_xy = cv::sum(I_x.mul(I_y))[0];
        double s_I_xt = cv::sum(I_x.mul(event_frame))[0];
        double s_I_yt = cv::sum(I_y.mul(event_frame))[0];

        cv::Mat M = (cv::Mat_<double>(2, 2) << s_I_xx, s_I_xy, s_I_xy, s_I_yy);
        cv::Mat b = (cv::Mat_<double>(2, 1) << s_I_xt, s_I_yt);

        cv::Mat v = -M.inv() * b;

        patch.flow_angle_ = std::atan2(v.at<double>(0, 0), v.at<double>(1, 0));
        patch.initialized_ = true;
    }

    void Tracker::eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg) {
        std::unique_lock <std::mutex> images_lock(images_mutex_);

        if (!got_first_image_) {
            LOG_EVERY_N(INFO, 20) << "Events dropped since no image present.";
            return;
        }

        if (sensor_size_.width <= 0)
            sensor_size_ = cv::Size(msg->width, msg->height);

        for (const dvs_msgs::Event &e : msg->events) {
            insertEventInSortedBuffer(e);
        }

    }

    void Tracker::imageCallback(const sensor_msgs::Image::ConstPtr &msg) {
        std::unique_lock <std::mutex> images_lock(images_mutex_);

        if (sensor_size_.width <= 0)
            sensor_size_ = cv::Size(msg->width, msg->height);

        cv_bridge::CvImagePtr cv_ptr;

        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
        }
        catch (cv_bridge::Exception &e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        // wait for the image after FLAGS_first_image_t
        if (cv_ptr->header.stamp.toSec() < FLAGS_first_image_t) {
            return;
        }

        images_.insert(std::make_pair(msg->header.stamp, cv_ptr->image.clone()));
    }

} // namespace
