'use client';

import { useState, useCallback } from 'react';
import FileUpload from '@/components/rfq/FileUpload';
import RFQFileList from '@/components/rfq/RFQFileList';
import RFQDetail from '@/components/rfq/RFQDetail';
import { RFQFileEntry } from '@/types/api';
import styles from './page.module.css';

export default function RFQPage() {
  const [files, setFiles] = useState<RFQFileEntry[]>([]);
  const [selectedFile, setSelectedFile] = useState<RFQFileEntry | null>(null);

  const handleFileUploaded = useCallback((entry: RFQFileEntry) => {
    setFiles(prev => [entry, ...prev]);
  }, []);

  const handleSelectFile = useCallback((file: RFQFileEntry) => {
    setSelectedFile(file);
  }, []);

  const handleBack = useCallback(() => {
    setSelectedFile(null);
  }, []);

  const handleDeleteFile = useCallback((id: string) => {
    setFiles(prev => prev.filter(f => f.id !== id));
    if (selectedFile?.id === id) {
      setSelectedFile(null);
    }
  }, [selectedFile]);

  return (
    <main className={styles.main}>
      <div className={styles.container}>
        {selectedFile ? (
          <RFQDetail file={selectedFile} onBack={handleBack} />
        ) : (
          <>
            <header className={styles.header}>
              <h1 className={styles.title}>RFQ Upload</h1>
              <p className={styles.tagline}>
                Upload Request for Quote files and find matching products
              </p>
            </header>

            <FileUpload onFileUploaded={handleFileUploaded} />

            {files.length > 0 && (
              <RFQFileList 
                files={files} 
                onSelect={handleSelectFile}
                onDelete={handleDeleteFile}
              />
            )}

            {files.length === 0 && (
              <div className={styles.emptyState}>
                <div className={styles.emptyIcon}>📄</div>
                <p>Upload a CSV, Excel, or PDF file to get started</p>
              </div>
            )}
          </>
        )}
      </div>
    </main>
  );
}

