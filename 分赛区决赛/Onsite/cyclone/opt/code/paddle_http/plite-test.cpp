#include "httplib.h"

void helloword(const httplib::Request &req, httplib::Response &rsp)
{
  std::cout << "httplib server recv a req:" << req.path.c_str() << std::endl;
  //Students fill in the blanks
  rsp.set_content("<html><h1> hello,world</h1></html>", "text/html");
  rsp.status = 200;
}
int main()
{
  httplib::Server srv;
  srv.Get("/hello",helloword);
  srv.Post("/upload", [](const httplib::Request &req, httplib::Response &res) {
  std::cout << "httplib server recv a req:" << req.path.c_str() << std::endl;
  const auto& file = req.get_file_value("image_file");
  std::cout << "filename:" << file.filename << std::endl;
  std::cout << "content_type:" << file.content_type << std::endl;
  res.set_content("success upload image", "text/plain");
  res.status = 200;
});
// listen port 9000
std::cout << "success run server" << std::endl;
srv.listen("0.0.0.0", 9000);

}
