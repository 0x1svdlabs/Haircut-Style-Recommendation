import subprocess
import sys

def main():
    while True:
        print("\n=============================================================")
        print("============== HairStyle Recommendation System ==============")
        print("=============================================================")
        print(" 1. Start HairStyle Recommendation")
        print(" 2. Pelatihan Model (Auto-sort -> Ekstrak -> Train -> Eval)")
        print(" 3. Lihat Akurasi AI Bentuk Wajah")
        print(" 4. Keluar")
        print("=============================================================")
        
        choice = input("Masukkan pilihan (1-4): ").strip()
        
        if choice == "1":
            print("\nMemulai HairStyle Recommendation...")
            print("INFO: Tekan [ESC] pada jendela kamera untuk keluar & kembali ke menu.\n")
            subprocess.run([sys.executable, "src/main.py"])
        elif choice == "2":
            print("\nMelatih Model...")
            print("\n[1/4] Mengklasifikasi & mengurutkan foto mentah...")
            subprocess.run([sys.executable, "training/classify_and_sort.py"])
            
            print("\n[2/4] Mengekstrak fitur wajah...")
            subprocess.run([sys.executable, "training/prepare_dataset.py"])
            
            print("\n[3/4] Melatih model AI...")
            subprocess.run([sys.executable, "training/train_model.py"])
            
            print("\n[4/4] Mengevaluasi model AI...")
            subprocess.run([sys.executable, "training/evaluate.py"])
            
            print("\nProses Pelatihan Selesai!")
        elif choice == "3":
            print("\nMengevaluasi Model...")
            subprocess.run([sys.executable, "training/evaluate.py"])
        elif choice == "4":
            print("\nKeluar... Sampai jumpa!")
            break
        else:
            print("\nPilihan tidak valid! Silakan masukkan angka 1-4.")

if __name__ == "__main__":
    main()
