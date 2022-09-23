#tf1.x_instance_segmentatioon
Projede kullanılan segmentasyon modeli tensorflow 1.x Object Detection API'sine ait mask_rcnn_inception_v2_coco_2018_01_28 modelidir. Modeli kulllanırken "https://github.com/tensorflow/models" github adresinden object detection kütüphanesinide kod içerisine dahil etmelisiniz.

Dizin Detayları:
- ./predict : Frozen graph dosyasına ait test kodu ve ön işlem kodunu bulunduruyor.
- ./predict_image : Test için sorulan örnek bir kaç resim dosyası barındırıyor.
- ./predict_image_response : Test için sorulan soruların modelden dönen segmentasyon ve bbox bilgisini barındıran resimler.
> (NOT: Segmentasyon haritalarının resimden daha net anlaşılması için bbox bilgiside eklendi.)

- ./pretrain_model : Transfer öğrenimi için kullanılan pretrain model dosyası içeriyor.
- ./re_pretrain_model: Transfer öğrenimi sonrasında kayıt edilen model dosyası içeriyor.
> (NOT: İlgili model dosyası:
**Saved_model formatı,**
**Frozen_graph formatı** ve **Checkpoint formatı **olacak şekilde verilmiştir.)

- ./main.py: Eğitime ön hazırlık aşama kodu olmaktadır. (İçerisinde veri seti ayırma,record dosyaları oluşturma ve eğitim kodu başlatma detayları yer almaktadır.)

------------


Eğitim-Test Detayları:
- Eğitim verisi %80-%20 oranında train ve test seti olarak ayrılmıştır.(Veri seti sayısı az olduğu için evaluation kısmı için veri seti ayrılmadı.)
- Eğitim verilerinin tf modeli üzerinde eğitimi için .record dosyalarını dönüştürülüp kullanılmaldır.(Oluşturulan record dosyaları 3.4GB boyutunda olduğu için githuba atılmadı.)
- Oluşturulan record dosyaları ile modele ait configürasyon dosyasında(pipeline.config) gerekli parametreler ayarlandı.(sınıf sayısı vb.)
- Eğitime hazır veri object detection kütüphanesine ait train.py kodu ile çalıştırıldı.(*tensorflow/models/tree/master/research/object_detection/legacy/train.py*)
- Eğitim devam ederken eğitilen modele ait kaydedilen checkpoint dosyalarından frozen graph elde edilerek model dosyası kaydedildi. Model dosyası kaydetme işlemi terminal ekranından yapılmıştır. (*tensorflow/models/tree/master/research/object_detection/export_inference_graph.py*)
- Kaydedilen model dosyası test resimleri ile test edildi. (predict/predict.py)

------------

Hazırlık Aşaması:
- virtualenv -p python3.6 py36
- source py36/bin/activate
- pip install -r requirements.txt
- python main.py (Dosya içerisinde gerekli veri seti pathleri doğru şekilde ayarlanarak çalıştırılmalıdır.)
