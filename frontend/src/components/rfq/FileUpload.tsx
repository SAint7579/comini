'use client';

import { useState, useRef, DragEvent } from 'react';
import { uploadRFQ } from '@/lib/api';
import { RFQFileEntry } from '@/types/api';
import styles from './FileUpload.module.css';

interface FileUploadProps {
  onFileUploaded: (entry: RFQFileEntry) => void;
}

export default function FileUpload({ onFileUploaded }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = async (e: DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      await processFile(files[0]);
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      await processFile(files[0]);
    }
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const processFile = async (file: File) => {
    // Validate file type
    const validTypes = ['.csv', '.xlsx', '.xls', '.pdf'];
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    
    if (!validTypes.includes(ext)) {
      setError('Invalid file type. Supported: CSV, Excel, PDF');
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      const result = await uploadRFQ(file);
      
      const entry: RFQFileEntry = {
        id: crypto.randomUUID(),
        filename: file.name,
        uploadedAt: new Date(),
        sourceType: result.source_type,
        totalItems: result.total_items,
        items: result.items,
      };
      
      onFileUploaded(entry);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className={styles.container}>
      <div
        className={`${styles.dropzone} ${isDragging ? styles.dragging : ''} ${isUploading ? styles.uploading : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.xlsx,.xls,.pdf"
          onChange={handleFileSelect}
          className={styles.input}
        />
        
        {isUploading ? (
          <>
            <div className={styles.spinner} />
            <p className={styles.text}>Processing file...</p>
          </>
        ) : (
          <>
            <div className={styles.icon}>📁</div>
            <p className={styles.text}>
              Drag & drop or <span className={styles.link}>browse</span>
            </p>
            <p className={styles.hint}>CSV, Excel, or PDF files</p>
          </>
        )}
      </div>
      
      {error && (
        <div className={styles.error}>{error}</div>
      )}
    </div>
  );
}

