Ta sẽ xây dựng mô hình dựa theo các mô hình GAN tích chập sâu (deep convolutional GAN - DCGAN) được giới thiệu trong [Radford et al., 2015]. Bằng cách mượn kiến trúc tích chập đã được chứng minh là thành công với bài toán thị giác máy tính phân biệt, và bằng cách thông qua GAN, ta có thể dùng chúng làm đòn bẩy để tạo ra các hình ảnh chân thực.
Sử dụng bộ dữ liệu Pokemon dùng làm ảnh gốc 

![image](https://user-images.githubusercontent.com/76995105/126550607-74be9031-1119-440d-9ae8-bfa1eb157903.png)

Kêt quả sau khi thực hiện train model :

![image](https://user-images.githubusercontent.com/76995105/126550704-086a3fa2-d38c-44ea-88c2-38b0803f879a.png)

Mô hình gồm hai phần là :
+ mạng sinh gồm các tầng tích chập chuyển vị 
+ mạng phân biệt là các tâng tích chập 
