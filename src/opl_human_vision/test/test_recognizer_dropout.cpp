#include "opl_human_vision/face_recognizer_node.hpp"
#include <cassert>
#include <thread>

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto recognizer = std::make_shared<opl_human_vision::FaceRecognizerNode>();
  recognizer->configure();
  recognizer->activate();
  auto probe = std::make_shared<rclcpp::Node>("recognizer_dropout_test");
  auto pub = probe->create_publisher<opl_interfaces::msg::TrackedHumanArray>("tracked_humans", rclcpp::SensorDataQoS());
  opl_interfaces::msg::TrackedHumanArray received;
  int count = 0;
  auto sub = probe->create_subscription<opl_interfaces::msg::TrackedHumanArray>(
    "recognized_humans", rclcpp::SensorDataQoS(),
    [&](opl_interfaces::msg::TrackedHumanArray::ConstSharedPtr msg) { received = *msg; ++count; });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(recognizer->get_node_base_interface());
  executor.add_node(probe);
  opl_interfaces::msg::TrackedHumanArray input;
  input.humans.resize(1);
  auto& h = input.humans.front();
  h.raw_track_id = h.track_id = 1; h.observed = true;
  h.association_ambiguous = true;
  auto exchange = [&]() {
    const int before = count;
    for (int i = 0; i < 100 && count == before; ++i) {
      input.header.stamp = probe->now();
      pub->publish(input);
      executor.spin_some();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      executor.spin_some();
    }
    assert(count > before && received.humans.size() == 1);
  };
  // No RGB stream: ambiguity must still be remembered by the recognizer.
  exchange();
  assert(received.humans.front().identity_state == h.UNCERTAIN);
  h.association_ambiguous = false;
  exchange();
  assert(received.humans.front().identity_state == h.UNCERTAIN);
  const int before_stale = count;
  input.header.stamp.sec = 1;
  input.header.stamp.nanosec = 0;
  pub->publish(input);
  for (int i = 0; i < 20; ++i) {
    executor.spin_some();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  assert(count == before_stale);
  auto client = probe->create_client<opl_interfaces::srv::EnrollPerson>("/face_recognizer_node/enroll_person");
  assert(client->wait_for_service(std::chrono::seconds(1)));
  auto request = std::make_shared<opl_interfaces::srv::EnrollPerson::Request>();
  request->raw_track_id = 1; request->name = "furia";
  auto accepted = client->async_send_request(request);
  assert(executor.spin_until_future_complete(accepted, std::chrono::seconds(1)) == rclcpp::FutureReturnCode::SUCCESS);
  assert(accepted.get()->success);
  request->raw_track_id = 99;
  auto rejected = client->async_send_request(request);
  assert(executor.spin_until_future_complete(rejected, std::chrono::seconds(1)) == rclcpp::FutureReturnCode::SUCCESS);
  assert(!rejected.get()->success);
  std::this_thread::sleep_for(std::chrono::milliseconds(800));
  request->raw_track_id = 1;
  auto expired = client->async_send_request(request);
  assert(executor.spin_until_future_complete(expired, std::chrono::seconds(1)) == rclcpp::FutureReturnCode::SUCCESS);
  assert(!expired.get()->success);
  recognizer->deactivate();
  recognizer->cleanup();
  rclcpp::shutdown();
}
