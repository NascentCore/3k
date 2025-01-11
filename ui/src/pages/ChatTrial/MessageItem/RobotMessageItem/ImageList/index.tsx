 
import React from 'react';
import { Image } from 'antd';
import { BaseUrl } from '@/pages/ChatTrial/services';

interface ImageListProps {
  show_images: string[];
}

const ImageList: React.FC<ImageListProps> = ({ show_images }) => {
  // 提取图片URLs，跳过第一个元素
  const imageUrls = show_images.slice(1).map((image) => {
    // 从markdown格式的字符串中提取URL
    const urlMatch = image.match(/!\[.*?\]\((.*?)\)/);
    return urlMatch ? urlMatch[1] : '';
  });
  return (
    <div className="image-list">
      <div style={{ marginBottom: 8 }}>引用图文如下：</div>
      {imageUrls.map((imageUrl, index) => (
        <div key={index} className="image-item">
          <Image
            src={`${BaseUrl}${imageUrl}`}
            alt={`Figure ${index}`}
            style={{ marginBottom: 6 }}
          />
        </div>
      ))}
    </div>
  );
};

export default ImageList;
